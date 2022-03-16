from matplotlib import pyplot as plt
from DataCollectionFunctions import *
from GeoImageFunctions import warp_geotiff, transform_raster_coordinate_by_raster


def to_save_image_as_geotiff(image, geo_trans, projection_ref, file_name, file_directory):
    # адрес сохраняемой карты
    file_address = "".join([file_directory, '/', file_name, '.tif'])
    # проверка количество каналов карты
    #image = image / np.max(image) * 255
    #if len(np.shape(image)) < 3:
    #    reshaped_image = np.array([image])
    #else:
    #    # переформатирование карты
    #    reshaped_image = np.moveaxis(image, -1, 0)
    reshaped_image = image
    bands_quantity = np.shape(reshaped_image)[0]
    # создание файла
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(file_address, len(image[0, 0]), len(image[0, :, 0]), bands_quantity,
                            gdal.GDT_Float32)  # gdal.GDT_Float32)
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
    del dataset

if __name__ == "__main__":
    shape_path = 'D:/Проекты/Велики Столак/Vector/Border_2.shp'
    dem_path = 'D:/Проекты/Велики Столак/Raster/output_COP30_32634.tif'
    sentinel_data_prefix = 'D:/Проекты/Велики Столак/Raster/Sentinel_2A 2021.05.09/' \
                           'S2B_MSIL2A_20210509T094029_N0300_R036_T34TCP_20210509T120133.SAFE/GRANULE/' \
                           'L2A_T34TCP_A021798_20210509T094028/'
    sentinel_path = sentinel_data_prefix + 'IMG_DATA/T34TCP_20210509T094029'
    mtp_file_path = sentinel_data_prefix + 'MTD_TL.xml'
    sentinel_B1_path = sentinel_path + '_B01.jp2'
    sentinel_B2_path = sentinel_path + '_B02.jp2'
    sentinel_B3_path = sentinel_path + '_B03.jp2'
    sentinel_B4_path = sentinel_path + '_B04.jp2'
    sentinel_B5_path = sentinel_path + '_B05.jp2'
    sentinel_B6_path = sentinel_path + '_B06.jp2'
    sentinel_B7_path = sentinel_path + '_B07.jp2'
    sentinel_B8_path = sentinel_path + '_B08.jp2'
    sentinel_B8A_path = sentinel_path + '_B8A.jp2'
    sentinel_B9_path = sentinel_path + '_B09.jp2'
    sentinel_B10_path =sentinel_path + '_B10.jp2'
    sentinel_B11_path = sentinel_path + '_B11.jp2'
    sentinel_B12_path = sentinel_path + '_B12.jp2'

    corr_path = 'D:/Проекты/Велики Столак/Raster'

    date = '2021.05.09'
    file_name_arr = ['DEM_32634']
    # file_name_arr = [
    #                  'Sentinel_B2_B3_B4_B8_10m_' + date,
    #                  'Sentinel_B5_B6_B7_B8A_B11_B12_20m_' + date,
    #                  'Sentinel_B1_B9_B10_60m_' + date,
    #                  ]
    # file_name_arr = [
    #                  'Sentinel_B1_60m_' + date,
    #                  'Sentinel_B2_10m_' + date,
    #                  'Sentinel_B3_10m_' + date,
    #                  'Sentinel_B4_10m_' + date,
    #                  'Sentinel_B5_20m_' + date,
    #                  'Sentinel_B6_20m_' + date,
    #                  'Sentinel_B7_20m_' + date,
    #                  'Sentinel_B8_10m_' + date,
    #                  'Sentinel_B8A_20m_' + date,
    #                  'Sentinel_B9_60m_' + date,
    #                  #'Sentinel_B10_60m_' + date,
    #                  'Sentinel_B11_20m_' + date,
    #                  'Sentinel_B12_20m_' + date
    #                  ]

    file_directory = 'D:/Проекты/Велики Столак/Raster/Sep Clipped Sentinel_2A ' + date
    # file_directory = 'D:/Проекты/Велики Столак/Raster/Clipped Sentinel ' + date
    # file_directory = 'D:/Проекты/Велики Столак/Raster/Sep Clipped Sentinel ' + date
    raster_path_arr = [dem_path]
    # raster_path_arr = [
    #                    #dem_path,
    #                    sentinel_B1_path,
    #                    sentinel_B2_path,
    #                    sentinel_B3_path,
    #                    sentinel_B4_path,
    #                    sentinel_B5_path,
    #                    sentinel_B6_path,
    #                    sentinel_B7_path,
    #                    sentinel_B8_path,
    #                    sentinel_B8A_path,
    #                    sentinel_B9_path,
    #                    #sentinel_B10_path,
    #                    sentinel_B11_path,
    #                    sentinel_B12_path
    #                    ]

    group_by = None
    #group_by = [[1, 2, 3, 7], [4, 5, 6, 8, 10, 11], [0, 9]]
    #group_by = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]
    if group_by is None:
        group_by = [np.arange(len(file_name_arr))]

    shape = shapefile.Reader(shape_path).shape(0)

    # warp_geotiff(dem_path, 'D:/Проекты/Велики Столак/Raster/output_COP30_32634.tif', 32634)
    # dem_rast = gdal.Open(dem_path)
    # dem_im = dem_rast.ReadAsArray()
    # dem_geo_trans = dem_rast.GetGeoTransform()
    # samp_rast_path = 'D:/Проекты/Велики Столак/Raster/Clipped Sentinel 2021.05.09/Sentinel_B8_10m_2021.05.04.tif'
    # samp_rast = gdal.Open(samp_rast_path)
    # samp_im = samp_rast.ReadAsArray()
    # samp_geo_trans = samp_rast.GetGeoTransform()
    # samp_projection_ref = samp_rast.GetProjectionRef()
    #im, im_geo_trans = transform_raster_coordinate_by_raster(dem_im, dem_geo_trans, samp_im, samp_geo_trans, samp_projection_ref)
    #to_save_image_as_geotiff(np.array([np.flipud(im)]), im_geo_trans, samp_projection_ref, 'new_DEM', file_directory)

    for i, group in enumerate(group_by):
        hypercube = []
        rast = gdal.Open(raster_path_arr[group[0]])
        geo_trans = rast.GetGeoTransform()
        projection_ref = rast.GetProjectionRef()
        new_geo_trans = None
        for j in group:
            rast = gdal.Open(raster_path_arr[j])
            image = np.float32(rast.ReadAsArray())
            clipped_image, new_geo_trans = to_clip_shot(image, shape, geo_trans, projection_ref, rectangle_shape=True, accurate_pol=True)
            hypercube.append(clipped_image)
        to_save_image_as_geotiff(np.array(hypercube), new_geo_trans, projection_ref, file_name_arr[i], file_directory)

    # mtd_file = minidom.parse(mtp_file_path)
    # time_data = mtd_file.getElementsByTagName('SENSING_TIME')
    # time_str = time_data[0].childNodes[0].data[:-1]
    # time_str = time_str[:10] + ' ' + time_str[11:]
    # time_str = time_str[:4] + '.' + time_str[5:]
    # time_str = time_str[:7] + '.' + time_str[8:]

    # accurate_sun = False
    # if not accurate_sun:
    #     sun_data_list = mtd_file.getElementsByTagName('Mean_Sun_Angle')
    #     zen_angle = sun_data_list[0].childNodes[1].childNodes[0].data
    #     az_angle = sun_data_list[0].childNodes[3].childNodes[0].data
    #     sun_data_file = open(file_directory + '/shot_data_' + date + '.txt', 'w')
    #     sun_data_file.write('DATE_TIME\t' + time_str + '\nZENITH_ANGLE\t' + zen_angle + '\nAZIMUTH_ANGLE\t' + az_angle)
    #     sun_data_file.close()
    # else:
    #     sun_data_list = mtd_file.getElementsByTagName('Sun_Angles_Grid')
    #     col_step = np.float32(sun_data_list[0].childNodes[1].childNodes[1].childNodes[0].data)
    #     row_step = np.float32(sun_data_list[0].childNodes[3].childNodes[1].childNodes[0].data)