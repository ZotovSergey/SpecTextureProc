import numpy as np
import osr
import copy
import shapefile
from osgeo import gdal
import tifffile as tiff
import skimage
from PIL import Image, ImageDraw
from itertools import product
from shapely import geometry


def to_clip_shot(image, polygon, geo_trans, projection_ref, rectangle_shape=False, origin=None, grad_count=None,
                 accurate_pol=True):
    # новая геопривязка для вырезанного снимка

    new_geo_trans = list(copy.deepcopy(geo_trans))
    # перевод долготы и широты в проекцию снимка
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(projection_ref)
    crsGeo = osr.SpatialReference()
    crsGeo.ImportFromEPSG(4326)
    transformer_in_shot_world_coordinates = osr.CoordinateTransformation(crsGeo, spatial_reference)
    polygons_borders_tiff_coords = transformer_in_shot_world_coordinates.TransformPoints(polygon.points)
    x_tiff_coords = np.array(polygons_borders_tiff_coords)[:, 0]
    y_tiff_coords = np.array(polygons_borders_tiff_coords)[:, 1]
    new_geo_trans[0] = np.round(min(x_tiff_coords) / geo_trans[1]) * geo_trans[1]
    # new_geo_trans[0] = np.round(min(x_tiff_coords))
    new_geo_trans[3] = np.round(max(y_tiff_coords) / geo_trans[1]) * geo_trans[1]
    # new_geo_trans[3] = np.round(max(y_tiff_coords))
    # перевод крайних координат в пиксели изображения
    polygons_border_pix = geometry.Polygon(coord2pix(polygon.points, geo_trans, projection_ref))
    pixels_angle = polygons_border_pix.bounds
    min_x_pix = int(pixels_angle[0])
    max_x_pix = int(pixels_angle[2])
    min_y_pix = int(pixels_angle[1])
    max_y_pix = int(pixels_angle[3])
    if origin is not None:
        x_diff = int((new_geo_trans[0] - origin[0]) / abs(new_geo_trans[1]))
        if x_diff > 0:
            # max_x_pix -= x_diff
            min_x_pix -= x_diff
            new_geo_trans[0] = origin[0]
            res_diff = origin[2] / geo_trans[1]
            max_x_pix = int(np.round(max_x_pix / res_diff) * res_diff)
        y_diff = -int((new_geo_trans[3] - origin[1]) / abs(new_geo_trans[1]))
        if y_diff > 0:
            # max_y_pix -= y_diff
            min_y_pix -= y_diff
            new_geo_trans[3] = origin[1]
            res_diff = origin[2] / geo_trans[1]
            max_y_pix = int(np.round(max_y_pix / res_diff) * res_diff)


    if len(image.shape) >= 3:
        new_image = copy.deepcopy(image[:, min_y_pix: max_y_pix, min_x_pix: max_x_pix])
        new_geo_trans = tuple(new_geo_trans)

        # если снимок должен быть прямоугольным
        if not rectangle_shape:
            if not accurate_pol:
                new_border_points = coord2pix(polygon.points, new_geo_trans, projection_ref)
                parts = copy.deepcopy(polygon.parts)
                parts.append(len(polygon.points))
                new_x_shot_size = len(new_image[0, 0])
                new_y_shot_size = len(new_image[0])

                mask = np.zeros((new_y_shot_size, new_x_shot_size))

                for i, end_part in enumerate(parts[1:]):
                    start_part = parts[i]
                    new_mask_im = Image.new('L', (new_x_shot_size, new_y_shot_size), 0)
                    ImageDraw.Draw(new_mask_im).polygon(new_border_points[start_part: end_part], outline=1, fill=1)
                    new_mask = np.array(new_mask_im)
                    inverse_inx = np.where(new_mask == 1)
                    mask[inverse_inx] = np.uint8(np.logical_not(mask[inverse_inx]))
                new_image = new_image * mask
            else:
                parts = copy.deepcopy(polygon.parts)
                parts.append(len(polygon.points) - 1)
                main_polygons_border_points_pix = polygon.points[parts[0]:parts[1]]
                polygons_by_parts = []
                for i in range(len(parts) - 1):
                    polygons_by_parts.append(
                        geometry.Polygon(coord2pix(polygon.points[parts[i]:parts[i + 1]], new_geo_trans,
                                                   projection_ref, to_round_result=False)))
                for i in range(len(new_image[0])):
                    for j in range(len(new_image[0, i])):
                        point = geometry.Point((j + 0.5, i + 0.5))
                        point_in_polygon = False
                        for pol in polygons_by_parts:
                            if pol.contains(point):
                                point_in_polygon = not point_in_polygon
                        if not point_in_polygon:
                            new_image[:, i, j] = np.nan
        if grad_count is not None:
            for i, im in enumerate(new_image):
                new_image[i] = to_transform_matrix_grey_grad(im, grad_count)
    else:
        new_image = copy.deepcopy(image[min_y_pix: max_y_pix, min_x_pix: max_x_pix])
        new_geo_trans = tuple(new_geo_trans)

        # если снимок должен быть прямоугольным
        if not rectangle_shape:
            if not accurate_pol:
                new_border_points = coord2pix(polygon.points, new_geo_trans, projection_ref)
                parts = copy.deepcopy(polygon.parts)
                parts.append(len(polygon.points))
                new_x_shot_size = len(new_image[0])
                new_y_shot_size = len(new_image)

                mask = np.zeros((new_y_shot_size, new_x_shot_size))

                for i, end_part in enumerate(parts[1:]):
                    start_part = parts[i]
                    new_mask_im = Image.new('L', (new_x_shot_size, new_y_shot_size), 0)
                    ImageDraw.Draw(new_mask_im).polygon(new_border_points[start_part : end_part], outline=1, fill=1)
                    new_mask = np.array(new_mask_im)
                    inverse_inx = np.where(new_mask == 1)
                    mask[inverse_inx] = np.uint8(np.logical_not(mask[inverse_inx]))
                new_image = new_image * mask
            else:
                parts = copy.deepcopy(polygon.parts)
                parts.append(len(polygon.points) - 1)
                main_polygons_border_points_pix = polygon.points[parts[0]:parts[1]]
                polygons_by_parts = []
                for i in range(len(parts) - 1):
                    polygons_by_parts.append(geometry.Polygon(coord2pix(polygon.points[parts[i]:parts[i + 1]], new_geo_trans,
                                                                        projection_ref, to_round_result=False)))
                for i in range(len(new_image)):
                    for j in range(len(new_image[i])):
                        point = geometry.Point((j + 0.5, i + 0.5))
                        point_in_polygon = False
                        for pol in polygons_by_parts:
                            if pol.contains(point):
                                point_in_polygon = not point_in_polygon
                        if not point_in_polygon:
                            new_image[i, j] = np.nan
        if grad_count is not None:
            new_image = to_transform_matrix_grey_grad(new_image, grad_count)
    return new_image, new_geo_trans

def to_make_mask(image_shape, polygons, geo_trans, projection_ref):
    mask = np.zeros(image_shape)

    for polygon in polygons:
        # перевод долготы и широты в проекцию снимка
        # перевод крайних координат в пиксели изображения
        polygons_border_pix = geometry.Polygon(coord2pix(polygon.points, geo_trans, projection_ref))
        pixels_angle = polygons_border_pix.bounds
        min_x_pix = int(pixels_angle[0])
        max_x_pix = int(pixels_angle[2])
        min_y_pix = int(pixels_angle[1])
        max_y_pix = int(pixels_angle[3])

        parts = copy.deepcopy(polygon.parts)
        parts.append(len(polygon.points) - 1)
        polygons_by_parts = []
        for i in range(len(parts) - 1):
            polygons_by_parts.append(
                geometry.Polygon(coord2pix(polygon.points[parts[i]:parts[i + 1]], geo_trans,
                                           projection_ref, to_round_result=False)))

        for x in range(min_x_pix, max_x_pix):
            for y in range(min_y_pix, max_y_pix):
                point = geometry.Point((x + 0.5, y + 0.5))
                point_in_polygon = False
                for pol in polygons_by_parts:
                    if pol.contains(point):
                        point_in_polygon = not point_in_polygon
                if point_in_polygon:
                    mask[y, x] = 1
    return mask

def mult_res_increase(image, multiplier):
    new_image = image
    new_image = np.repeat(new_image, multiplier, axis=1)
    new_image = np.repeat(new_image, multiplier, axis=0)
    return new_image
    # new_image_list = []
    # new_im = None
    # max_res = min(res_list)
    # max_res_inx = np.where(np.array(res_list) == max_res)
    # max_res_image = np.array(image_list)[max_res_inx][0]
    # max_res_geo_trans = np.array(geo_trans_list)[max_res_inx][0]
    # max_res_proj_ref = np.array(projection_ref_list)[max_res_inx][0]
    # for i, im in enumerate(image_list):
    #     new_im = im
    #     if res_list[i] > max_res:
    #         new_im = np.zeros(max_res_image.shape)
    #         new_im_pix = np.float64(np.array(list(product(range(max_res_image.shape[0]), range(max_res_image.shape[1]))))) + 0.5
    #         new_im_coord = np.array(pix2coord(new_im_pix, max_res_geo_trans, max_res_proj_ref))
    #         new_im_coord_x = new_im_coord[:, 0]
    #         new_im_coord_y = new_im_coord[:, 1]
    #         old_image = image_list[i]
    #         for p in range(len(old_image)):
    #             for q in range(len(old_image[p])):
    #                 old_pix_angles_coord = pix2coord(np.array([[p, q], [p + 1, q + 1]]), geo_trans_list[i], projection_ref_list[i])
    #                 new_pix_in_old_inx = np.where((new_im_coord_x >= old_pix_angles_coord[0][0]) &
    #                                               (new_im_coord_x < old_pix_angles_coord[1][0]) &
    #                                               (new_im_coord_y < old_pix_angles_coord[0][1]) &
    #                                               (new_im_coord_y >= old_pix_angles_coord[1][1]))[0]
    #                 rows = np.int32(np.floor(new_pix_in_old_inx / max_res_image.shape[1]))
    #                 cols = new_pix_in_old_inx % max_res_image.shape[1]
    #                 new_im[rows, cols] = old_image[p][q]
    #     new_image_list.append(new_im)
    # return new_image_list


def coord2pix(coordinates_list, geo_trans, projection_ref, to_round_result=True):
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(projection_ref)
    crsGeo = osr.SpatialReference()
    crsGeo.ImportFromEPSG(4326)
    transformer_in_shot_world_coordinates = osr.CoordinateTransformation(crsGeo, spatial_reference)
    coords = transformer_in_shot_world_coordinates.TransformPoints(coordinates_list)
    if len(coords) > 0:
        x_coords = np.array(coords)[:, 0]
        y_coords = np.array(coords)[:, 1]

        x_pix = (-x_coords * geo_trans[5] + y_coords * geo_trans[2] +
                 geo_trans[0] * geo_trans[5] - geo_trans[2] * geo_trans[3]) / \
                (geo_trans[2] * geo_trans[4] - geo_trans[1] * geo_trans[5])
        y_pix = (x_coords * geo_trans[4] - y_coords * geo_trans[1] +
                 geo_trans[1] * geo_trans[3] - geo_trans[0] * geo_trans[4]) / \
                 (geo_trans[2] * geo_trans[4] - geo_trans[1] * geo_trans[5])
        if to_round_result:
            x_pix = np.round(x_pix)
            y_pix = np.round(y_pix)
        output = list(zip(x_pix, y_pix))
    else:
        output = []
    return output


def pix2coord(pix_coordinates_list, geo_trans, projection_ref, own_projection_ref=False):
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection_ref)
    ct = osr.CoordinateTransformation(srs, srs.CloneGeogCS())
    if len(pix_coordinates_list) > 0:
        x = np.array(pix_coordinates_list)[:, 0]
        y = np.array(pix_coordinates_list)[:, 1]
        lon_list = x * geo_trans[1] + geo_trans[0]
        lat_list = y * geo_trans[5] + geo_trans[3]
        if own_projection_ref:
            lon_final_list = lon_list
            lat_final_list = lat_list
        else:
            lon_final_list = []
            lat_final_list = []
            for i in range(0, len(lon_list)):
                (lon, lat, holder) = ct.TransformPoint(lon_list[i], lat_list[i])
                lon_final_list.append(lon)
                lat_final_list.append(lat)
        output = list(zip(lon_final_list, lat_final_list))
    else:
        output = []
    return output


def to_transform_image_grey_grad(image_address, new_image_directory, new_image_name, grad_count):
    # Чтение изображения
    matrix = tiff.imread(image_address)
    # Чтение привязки
    rast = gdal.Open(image_address)
    geo_trans = rast.GetGeoTransform()
    projection_ref = rast.GetProjectionRef()

    graded_matrix = to_transform_matrix_grey_grad(matrix, grad_count)

    # адрес сохраняемой карты
    new_file_address = "".join([new_image_directory, '/', new_image_name, '.tif'])
    # проверка количество каналов карты
    if len(np.shape(graded_matrix)) < 3:
        reshaped_image = np.array([graded_matrix])
    else:
        # переформатирование карты
        reshaped_image = np.moveaxis(graded_matrix, -1, 0)
    bands_quantity = np.shape(reshaped_image)[0]
    # создание файла
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(new_file_address, len(graded_matrix[0]), len(graded_matrix[:, 0]), bands_quantity, gdal.GDT_Byte)
    dataset.SetGeoTransform((geo_trans[0],
                             geo_trans[1],
                             geo_trans[2],
                             geo_trans[3],
                             geo_trans[4],
                             geo_trans[5]))
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(projection_ref)
    dataset.SetProjection(spatial_reference.ExportToWkt())
    for i in range(0, bands_quantity):
        dataset.GetRasterBand(i + 1).WriteArray(reshaped_image[i])
    dataset.FlushCache()
    dataset = None

    matrix1= tiff.imread(new_file_address)
    print()

def to_transform_and_clip_image_grey_grad(image_address, new_image_directory, new_image_name, grad_count,
                                          border_shape_address, rectangle_shape=False):
    # Чтение изображения
    rast = gdal.Open(image_address)
    matrix = []
    for i in range(1, rast.RasterCount + 1):
        matrix.append(rast.GetRasterBand(i).ReadAsArray())
    matrix = np.array(matrix)
    # Чтение привязки
    geo_trans = rast.GetGeoTransform()
    projection_ref = rast.GetProjectionRef()
    border_shape = shapefile.Reader(border_shape_address).shapes()[0]

    graded_matrix, geo_trans = to_clip_shot(matrix, border_shape, geo_trans, projection_ref,
                                            rectangle_shape=rectangle_shape, grad_count=grad_count)

    # адрес сохраняемой карты
    new_file_address = "".join([new_image_directory, '/', new_image_name, '.tif'])
    # проверка количество каналов карты
    #if len(np.shape(graded_matrix)) < 3:
    #    reshaped_image = np.array([graded_matrix])
    #else:
    #    # переформатирование карты
    #    reshaped_image = np.moveaxis(graded_matrix, -1, 0)
    bands_quantity = np.shape(graded_matrix)[0]
    # создание файла
    driver = gdal.GetDriverByName('GTiff')
    #dataset = driver.Create(new_file_address, graded_matrix.shape[2], graded_matrix.shape[1], bands_quantity, gdal.GDT_Byte)
    dataset = driver.Create(new_file_address, graded_matrix.shape[2], graded_matrix.shape[1], bands_quantity, gdal.GDT_Byte)
    dataset.SetGeoTransform((geo_trans[0],
                             geo_trans[1],
                             geo_trans[2],
                             geo_trans[3],
                             geo_trans[4],
                             geo_trans[5]))
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(projection_ref)
    dataset.SetProjection(spatial_reference.ExportToWkt())
    for i in range(0, bands_quantity):
        dataset.GetRasterBand(i + 1).WriteArray(graded_matrix[i, :, :])
    dataset.FlushCache()
    dataset = None

    matrix1= tiff.imread(new_file_address)
    #tiff.imsave(new_file_address, matrix1)
    print()

def to_transform_matrix_grey_grad(matrix, grad_count_borders):
    if len(grad_count_borders) > 1:
        low_border = grad_count_borders[0]
        grad_range = grad_count_borders[1] - low_border
    else:
        low_border = 1
        grad_range = grad_count_borders[0]
    min_val = np.min(matrix[np.nonzero(matrix)])
    max_val = np.max(matrix)
    trans_coef = grad_range / (max_val - min_val)
    graded_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
    calc_inx = np.where(matrix != 0)
    graded_matrix[calc_inx] = np.int32(np.round((matrix[calc_inx] - min_val) * trans_coef) + low_border)
    return np.int32(graded_matrix)

def warp_geotiff(imput_image_path, output_image_path, epsg_num):
    input_raster = gdal.Open(imput_image_path)
    epsg_str = 'EPSG:' + str(epsg_num)
    warp = gdal.Warp(output_image_path, input_raster, dstSRS=epsg_str)
    warp = None

def transform_raster_coordinate_by_raster(image, geo_trans, image_samp, geo_trans_samp, proj_ref):
    new_image = np.zeros(image_samp.shape)
    new_geo_trans = geo_trans_samp
    inx_list = np.float64(np.array(list(product(range(new_image.shape[0]), range(new_image.shape[1]))))) + 0.5
    init_x_inx_list = np.float64(np.array(list(range(image.shape[1] + 1))))
    init_y_inx_list = np.float64(np.array(list(range(image.shape[0] + 1))))
    init_x_inx_list = np.float64(np.array(list(zip(init_x_inx_list, np.zeros(len(init_x_inx_list))))))
    init_y_inx_list = np.float64(np.array(list(zip(np.zeros(len(init_y_inx_list)), init_y_inx_list))))
    init_x_coord_list = np.array(pix2coord(init_x_inx_list, geo_trans, proj_ref, own_projection_ref=True))[:, 0]
    init_y_coord_list = np.array(pix2coord(init_y_inx_list, geo_trans, proj_ref, own_projection_ref=True))[:, 1]
    coord_list = pix2coord(inx_list, new_geo_trans, proj_ref, own_projection_ref=True)
    inx_matrix = np.reshape(inx_list, (new_image.shape[0], new_image.shape[1], 2))
    coord_matrix = np.reshape(coord_list, (new_image.shape[0], new_image.shape[1], 2))

    for i in range(len(coord_matrix)):
        for j in range(len(coord_matrix[0])):
            p = 0
            while True:
                if (coord_matrix[i, j][0] >= init_x_coord_list[p]) and (coord_matrix[i, j][0] < init_x_coord_list[p + 1]):
                    break
                p += 1
            q = 0
            while True:
                if (coord_matrix[i, j][1] < init_y_coord_list[q]) and (coord_matrix[i, j][1] >= init_y_coord_list[q + 1]):
                    break
                q += 1
            new_image[int(inx_matrix[i, j][0] - 0.5), int(inx_matrix[i, j][1] - 0.5)] = image[p, q]

    return np.flipud(new_image), new_geo_trans