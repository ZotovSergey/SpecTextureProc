import pandas as pd
import re
import gdal
from xml.dom import minidom

from TextureCalc import to_calc_textures
from GeoImageFunctions import *

from Constants import TEXTURE_NAMES_LIST



def to_get_spectral_data_from_landsat8(band_names, band_keys, landsat8_shot_metadata_address):
    # Данные о каналах Landsat 8
    overall_band_data_frame = pd.DataFrame(
        {'band_name': ['BAND_1', 'BAND_2', 'BAND_3', 'BAND_4', 'BAND_5', 'BAND_6', 'BAND_7', 'BAND_8', 'BAND_9', 'BAND_10', 'BAND_11'],
         'upper_wavelength': [453, 515, 600, 680, 885, 1660, 2300, 680, 1390, 11300, 12500],
         'lower_wavelength': [433, 450, 525, 630, 845, 1560, 2100, 500, 1360, 10300, 11500],
         'spacial_resolution': [30, 30, 30, 30, 30, 30, 30, 15, 30, 100, 100]})
    # Чтение файла с методанными снимка
    landsat8_metadata_file = open(landsat8_shot_metadata_address, "r")
    # чтение данных из файла методанных
    metadata = landsat8_metadata_file.read()
    # закрытие файла с методанными
    landsat8_metadata_file.close()
    # DataFrame, содержащий данные о извлеченных каналах
    band_data_frame = pd.DataFrame(columns=['band_key',
                                            'band_name',
                                            'band_layer',
                                            'band_address',
                                            'upper_wavelength',
                                            'lower_wavelength',
                                            'central_wavelength',
                                            'spacial_resolution',
                                            'calibration_coef'])
    # Извлечение данных о каждом канале из band_names и их запись в видеобъекта Band в библиотеку spectral_data_dict
    #   под соответствующим ключом из band_keys
    for i, band_name in enumerate(band_names):
        band_interior_address = re.findall('"[\S]+"', re.search("".join(['FILE_NAME_', band_name, ' = "[\S]+"']), metadata).group())[0][1:-1]
        band_address = "/".join(["/".join(landsat8_shot_metadata_address.split('/')[0: -1]), band_interior_address])
        band_id = np.where(overall_band_data_frame['band_name'].values == band_name)[0][0]
        upper_wavelength = overall_band_data_frame['upper_wavelength'].values[band_id]
        lower_wavelength = overall_band_data_frame['lower_wavelength'].values[band_id]
        central_wavelength = (upper_wavelength + lower_wavelength) / 2
        space_res = overall_band_data_frame['spacial_resolution'].values[band_id]
        # запись данных в DataFrame
        band_data_frame = band_data_frame.append({'band_key': band_keys[i],
                                                  'band_name': band_names[i],
                                                  'band_layer': 0,
                                                  'band_address': band_address,
                                                  'upper_wavelength': upper_wavelength,
                                                  'central_wavelength': central_wavelength,
                                                  'lower_wavelength': lower_wavelength,
                                                  'spacial_resolution': space_res,
                                                  'calibration_coef': 1}, ignore_index=True)
    return band_data_frame

def to_get_spectral_data_from_sentinel2(band_names, band_keys, sentinel2_shot_metadata_address):
    # Чтение xml-файла с методанными снимка
    sentinel2_metadata_file = minidom.parse(sentinel2_shot_metadata_address)
    # данные о каналах из методанных
    spec_bands_xml_list = sentinel2_metadata_file.getElementsByTagName('Spectral_Information')
    # DataFrame, содержащий данные о извлеченных каналах
    band_data_frame = pd.DataFrame(columns=['band_key',
                                            'band_name',
                                            'band_layer',
                                            'band_address',
                                            'upper_wavelength',
                                            'lower_wavelength',
                                            'central_wavelength',
                                            'spacial_resolution',
                                            'calibration_coef'])
    # Извлечение данных о каждом канале из band_names и их запись в видеобъекта Band в библиотеку spectral_data_dict
    #   под соответствующим ключом из band_keys
    for i in range(0, len(band_names)):
        band_id = 0
        while (band_id < len(spec_bands_xml_list)) and \
                (spec_bands_xml_list[band_id]._attrs['physicalBand'].value != band_names[i]):
            band_id += 1
        band_interior_address = ".".join(
            [sentinel2_metadata_file.getElementsByTagName('Granule')[0].childNodes[
                 band_id * 2 + 1].firstChild.wholeText,
             "jp2"])
        band_address = "/".join(["/".join(sentinel2_shot_metadata_address.split('/')[0: -1]), band_interior_address])
        upper_wavelength = float(spec_bands_xml_list[band_id].childNodes[3].childNodes[3].firstChild.data)
        lower_wavelength = float(spec_bands_xml_list[band_id].childNodes[3].childNodes[5].firstChild.data)
        central_wavelength = (upper_wavelength + lower_wavelength) / 2
        space_res = float(spec_bands_xml_list[band_id].childNodes[1].firstChild.data)
        # запись данных в DataFrame
        band_data_frame = band_data_frame.append({'band_key': band_keys[i],
                                                  'band_name': band_names[i],
                                                  'band_layer': 0,
                                                  'band_address': band_address,
                                                  'upper_wavelength': upper_wavelength,
                                                  'central_wavelength': central_wavelength,
                                                  'lower_wavelength': lower_wavelength,
                                                  'spacial_resolution': space_res,
                                                  'calibration_coef': 1}, ignore_index=True)
    return band_data_frame

def to_get_spectral_data_from_worldview2(band_names, band_keys, worldview2_shot_metadata_address):
    # Данные о каналах WorldView2
    overall_band_data_frame = pd.DataFrame({'band_name': ['BAND_C', 'BAND_B', 'BAND_G', 'BAND_Y', 'BAND_R', 'BAND_RE', 'BAND_N', 'BAND_N2'],
                                            'band_layer': [0, 1, 2, 3, 4, 5, 6, 7],
                                            'upper_wavelength': [450, 510, 580, 625, 690, 745, 895, 1040],
                                            'lower_wavelength': [400, 450, 510, 585, 630, 705, 770, 860],
                                            'central_wavelength': [427, 478, 546, 608, 659, 724, 831, 908],
                                            'spacial_resolution': [2, 2, 2, 2, 2, 2, 2, 2]})
    # Чтение xml-файла с методанными снимка
    worldview2_metadata_file = minidom.parse(worldview2_shot_metadata_address)
    # DataFrame, содержащий данные о извлеченных каналах
    band_data_frame = pd.DataFrame(columns=['band_key',
                                            'band_name',
                                            'band_layer',
                                            'band_address',
                                            'upper_wavelength',
                                            'lower_wavelength',
                                            'central_wavelength',
                                            'spacial_resolution',
                                            'calibration_coef',
                                            'calibration_add'])
    # Адрес tiff-файла с каналами
    band_address = ".".join(["/".join(["/".join(worldview2_shot_metadata_address.split('/')[0: -1]),
                                       worldview2_shot_metadata_address.split('/')[-1].split('.')[0]]), 'TIF'])
    # Извлечение данных о каждом канале из band_names и их запись в видеобъекта Band в библиотеку spectral_data_dict
    #   под соответствующим ключом из band_keys
    for i, band_name in enumerate(band_names):
        band_id = np.where(overall_band_data_frame['band_name'].values == band_name)[0][0]
        band_layer = overall_band_data_frame['band_layer'].values[band_id]
        upper_wavelength = overall_band_data_frame['upper_wavelength'].values[band_id]
        lower_wavelength = overall_band_data_frame['lower_wavelength'].values[band_id]
        central_wavelength = overall_band_data_frame['central_wavelength'].values[band_id]
        space_res = overall_band_data_frame['spacial_resolution'].values[band_id]

        abs_cal_factor = float(worldview2_metadata_file.getElementsByTagName(band_name)[0].childNodes[25].firstChild.data)
        effective_bandwidth = float(worldview2_metadata_file.getElementsByTagName(band_name)[0].childNodes[27].firstChild.data)
        calibration_coef = abs_cal_factor / effective_bandwidth
        # запись данных в DataFrame
        band_data_frame = band_data_frame.append({'band_key': band_keys[i],
                                                  'band_name': band_name,
                                                  'band_layer': band_layer,
                                                  'band_address': band_address,
                                                  'upper_wavelength': upper_wavelength,
                                                  'lower_wavelength': lower_wavelength,
                                                  'central_wavelength': central_wavelength,
                                                  'spacial_resolution': space_res,
                                                  'calibration_coef': calibration_coef,
                                                  'calibration_add': 0}, ignore_index=True)
    return band_data_frame

def to_get_spectral_data_from_images(band_names, band_keys, shot_metadata_addresses, layers=None,
                                     upper_wavelengths=None, lower_wavelengths=None, calibration_coefficient=None,
                                     calibration_add_coefficient=None):
    # DataFrame, содержащий данные о извлеченных каналах
    band_data_frame = pd.DataFrame(columns=['band_key',
                                            'band_name',
                                            'band_layer',
                                            'band_address',
                                            'upper_wavelength',
                                            'lower_wavelength',
                                            'central_wavelength',
                                            'spacial_resolution',
                                            'calibration_coef',
                                            'calibration_add'])
    if layers is None:
        layers = len(band_keys) * [0]
    if upper_wavelengths is None:
        upper_wavelengths = len(band_keys) * [0]
    if lower_wavelengths is None:
        lower_wavelengths = len(band_keys) * [0]
    if calibration_coefficient is None:
        calibration_coefficient = len(band_keys) * [1]
    if calibration_add_coefficient is None:
        calibration_add_coefficient = len(band_keys) * [0]
    # Извлечение данных о каждом канале из band_names и их запись в видеобъекта Band в библиотеку spectral_data_dict
    #   под соответствующим ключом из band_keys
    for i in range(0, len(band_names)):
        band_address = shot_metadata_addresses[i]
        band_layer = layers[i]
        upper_wavelength = upper_wavelengths[i]
        lower_wavelength = lower_wavelengths[i]
        central_wavelength = (upper_wavelength + lower_wavelength) / 2
        space_res = gdal.Open(band_address).GetGeoTransform()[1]
        calibration_coef = calibration_coefficient[i]
        calibration_add = calibration_add_coefficient[i]
        # запись данных в DataFrame
        band_data_frame = band_data_frame.append({'band_key': band_keys[i],
                                                  'band_name': band_names[i],
                                                  'band_layer': band_layer,
                                                  'band_address': band_address,
                                                  'upper_wavelength': upper_wavelength,
                                                  'central_wavelength': central_wavelength,
                                                  'lower_wavelength': lower_wavelength,
                                                  'spacial_resolution': space_res,
                                                  'calibration_coef': calibration_coef,
                                                  'calibration_add': calibration_add}, ignore_index=True)
    return band_data_frame

def to_get_textures_from_image(texture_image, polygon, texture_geo_trans, texture_projection_ref,
                               spec_geo_trans=None, spec_projection_ref=None, spec_polygon_shape=None,
                               texture_list=TEXTURE_NAMES_LIST,
                               texture_adjacency_directions=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4), distance=1,
                               window_width=None, rectangle_shape=True, texture_data_linked_to_spec=False, grad_count=None,
                               accurate_pol=False):
    texture_data_dict_list = []
    new_spec_geo_trans_list = []
    new_texture_geo_trans_list = []
    # Замена начала координат в новом изображении. Применяется при привязке текстурных признаков к спектральным
    if texture_data_linked_to_spec:
        origin = [spec_geo_trans[0], spec_geo_trans[3], abs(spec_geo_trans[1])]
    else:
        origin = None

    # вырезание изображения по заданной форме polygon
    if polygon is not None:
        clipped_texture_im, new_texture_geo_trans = to_clip_shot(texture_image,
                                                                 polygon,
                                                                 texture_geo_trans,
                                                                 texture_projection_ref,
                                                                 rectangle_shape=rectangle_shape,
                                                                 origin=origin,
                                                                 grad_count=grad_count,
                                                                 accurate_pol=accurate_pol)
    else:
        clipped_texture_im = texture_image
        new_texture_geo_trans = texture_geo_trans
    # пространственное разрешение данных
    texture_space_res = abs(new_texture_geo_trans[1])
    # вычисление ширины и полуширины окна
    window = window_width
    if texture_data_linked_to_spec:
        spec_space_res = abs(spec_geo_trans[1])
        if window is None:
            window = round(spec_space_res / texture_space_res)
    # if window % 2 == 0:
    #     window -= 1
    half_window_width = int(window / 2)
    # вычисление координат пикселей спектральных данных через пиксели текстурных данных
    # координаты центров пикселей спектральных данных в собственной системе координат
    geo_centers = []
    if texture_data_linked_to_spec:
        new_spec_geo_trans = spec_geo_trans
        x_pix = np.arange(0, spec_polygon_shape[1])
        y_pix = np.arange(0, spec_polygon_shape[0])
        pix_of_spec_data = []
        for i in y_pix:
            for j in x_pix:
                pix_of_spec_data.append([x_pix[j], y_pix[i]])
        pix_of_spec_data = np.array(pix_of_spec_data) + 0.5
        # географические координаты центров пикселей спектральных данных
        geo_centers = pix2coord(pix_of_spec_data, spec_geo_trans, spec_projection_ref)
    else:
        new_spec_geo_trans = copy.deepcopy(list(new_texture_geo_trans))
        new_spec_geo_trans[1] *= window
        new_spec_geo_trans[5] *= window
        x_pix = np.arange(0, clipped_texture_im.shape[2], window)
        y_pix = np.arange(0, clipped_texture_im.shape[1], window)
        pix_of_text_data = []
        for i in np.arange(0, len(y_pix)):
            for j in np.arange(0, len(x_pix)):
                pix_of_text_data.append([x_pix[j], y_pix[i]])
        pix_of_text_data = np.array(pix_of_text_data) + half_window_width
        geo_centers = pix2coord(pix_of_text_data, new_texture_geo_trans, texture_projection_ref)
    # удаление координат, не входящих в контур
    if not rectangle_shape:
        pol = geometry.Polygon(polygon.points)
        for i in range(len(geo_centers)):
            point = geometry.Point(geo_centers[i])
            if not pol.contains(point):
                geo_centers[i] = (np.inf, np.inf)
    # координаты центров пикселей спектральных данных в системе координат текстурных данных
    pix_of_spec_data_in_texture_data = coord2pix(geo_centers, new_texture_geo_trans, texture_projection_ref)
    # массив координат центров пикселей спектральных данных в системе координат текстурных данных
    pix_of_spec_data_in_texture_array = np.zeros((len(y_pix), len(x_pix), 2))
    k = 0
    for i in range(len(y_pix)):
        for j in range(len(x_pix)):
            pix_of_spec_data_in_texture_array[i, j] = pix_of_spec_data_in_texture_data[k]
            k += 1
        # pix_of_spec_data_in_texture_array = np.array(pix_of_spec_data_in_texture_data).reshape(
        #     (spec_polygon_shape[0], spec_polygon_shape[1], 2))
    # нулевой массив, в который будут записываться текстурные данные
    texture_zero_array = np.zeros((len(y_pix), len(x_pix), len(distance), len(texture_adjacency_directions)))
    for s in range(len(texture_image)):
        # формирование библиотеки текстурных данных
        texture_data_dict = {}
        for texture in texture_list:
            texture_data_dict.update({texture: copy.deepcopy(texture_zero_array)})
        for i in range(0, texture_zero_array.shape[0]):
            for j in range(0, texture_zero_array.shape[1]):
                if not np.isnan(pix_of_spec_data_in_texture_array[i, j][0]):
                    print(i, j)
                    window_center = np.int32(pix_of_spec_data_in_texture_array[i, j])
                    window_center_x = window_center[0]
                    window_center_y = window_center[1]
                    left_border = window_center_x - half_window_width
                    right_border = window_center_x + half_window_width
                    up_border = window_center_y - half_window_width
                    down_border = window_center_y + half_window_width
                    if left_border < 0:
                        left_border = 0
                    if up_border < 0:
                        up_border = 0
                    window_mat = clipped_texture_im[s, up_border: down_border, left_border: right_border]
                    current_textures = to_calc_textures(window_mat, texture_adjacency_directions, texture_list, dist=distance)
                else:
                    current_textures = {}
                    for texture_name in texture_list:
                        current_textures.update({texture_name: np.zeros((len(distance), len(texture_adjacency_directions)))})
                for k, texture_name in enumerate(texture_list):
                    texture_data_dict[texture_list[k]][i][j] = current_textures[texture_name]
        for texture_name in texture_list:
            texture_data_dict[texture_name][np.where(np.isnan(texture_data_dict[texture_name]))] = 0
        texture_data_dict_list.append(texture_data_dict)
    return texture_data_dict_list, clipped_texture_im, new_spec_geo_trans, new_texture_geo_trans