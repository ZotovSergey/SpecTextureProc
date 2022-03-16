import pickle
from osgeo import gdal
import tifffile as tiff
import shapefile
import skimage
from matplotlib import colors
from matplotlib import pyplot as plt

from DataCollectionFunctions import *
from GeoImageFunctions import *


MAX_8_BIT = 255
TEXTURE_NAMES_LIST = ['Autocorrelation', 'ClusterProminence', 'ClusterShade', 'Contrast', 'Correlation', 'DiffEntropy',
                      'DiffVariance', 'Dissimilarity', 'Energy', 'Entropy', 'Homogeneity', 'Homogeneity2',
                      'InfMeasureCorr1', 'InfMeasureCorr2', 'MaxProb', 'SumAverage', 'SumEntropy', 'SumSquares',
                      'SumVariance']


def to_load_data_shot(save_address):
    """
    @ Описание:
        Метод возвращает объект Task с данными о задаче, пролетах, выполнении задач, сохраненный по адресу save_address.
    :param save_address: адресс файла, из которого загружается сохраненный объект OutputDataMaker (String).
    :return: объект класса OutputDataMaker, содержащий данные о задаче, пролетах, выполнении задач, прочитанные из файла
        по адресу save_address
    """
    with open(save_address, "rb") as file:
        loaded_data = pickle.load(file)
    return loaded_data


class DataShot:
    def __init__(self, name, data_shot_shape=None, rectangular_data_shot_shape=False):
        # Название снимка
        self.name = name
        # shape-файл, задающий форму снимка
        if type(data_shot_shape) == str:
            self.data_shot_shape = shapefile.Reader(data_shot_shape).shapes()[0]
        else:
            self.data_shot_shape = data_shot_shape
        # индикатор прямоугольности снимка
        self.rectangular_data_shot_shape = rectangular_data_shot_shape
        # максимальное пространственное разрешение спектральных данных
        self.spec_space_res = None
        # геопривязка спектрального снимка
        self.spec_geo_trans = None
        self.spec_projection_ref = None
        # библиотека спектральных данных
        self.spectral_data_dict = {}
        # пространственное разрешение текстурных данных
        self.texture_space_res = None
        # направления смежности, по которым вычислялись текстурные признаки
        self.texture_adjacency_directions = None
        # дистанция, по которым вычислялись текстурные признаки
        self.distances = None
        # геопривязка текстурного снимка
        self.texture_geo_trans = None
        self.texture_projection_ref = None
        # размер снимка в пикселях (соответствует спектральным данным)
        self.data_shape = None
        # библиотека текстурных данных
        self.texture_data_dict = {}
        # изображение снимка в RGB по спектральным данным
        self.rgb_image = None
        # изображение снимка в высоком разрешении
        self.high_res_image = None
        # изображение снимка в высоком разрешении
        self.grad_high_res_image = None

    def to_add_spectral_data_from_landsat8(self, band_names, band_keys, landsat8_shot_metadata_address):
        # Чтение адресов изображений сниимка Landsat-8 в заданных каналах (библиотека)
        band_table = to_get_spectral_data_from_landsat8(band_names,
                                                        band_keys,
                                                        landsat8_shot_metadata_address)
        # максимальное пространственное разрешение
        self.spec_space_res = min(list(band_table['spacial_resolution']))
        # список пространственных разрешений
        spec_space_res_list = []
        for band_key in band_keys:
            band_row = band_table[band_table['band_key'] == band_key]
            band_name = list(band_row['band_name'])[0]
            band_address = list(band_row['band_address'])[0]
            upper_wavelength = list(band_row['upper_wavelength'])[0]
            lower_wavelength = list(band_row['lower_wavelength'])[0]
            central_wavelength = list(band_row['central_wavelength'])[0]
            space_res = list(band_row['spacial_resolution'])[0]
            spec_space_res_list.append(space_res)
            # создание массивов спектральных данных по форме self.data_shape (объект Band)
            band = Band(band_name, band_address, space_res, upper_wavelength, lower_wavelength, central_wavelength,
                        self.data_shot_shape, rectangle_shape=self.rectangular_data_shot_shape,
                        multiplier=int(space_res / self.spec_space_res))
            self.spectral_data_dict.update({band_key: band})
        self.spec_geo_trans = self.spectral_data_dict[band_keys[0]].geo_trans
        self.spec_projection_ref = self.spectral_data_dict[band_keys[0]].projection_ref
        self.data_shape = self.spectral_data_dict[band_keys[0]].band.shape

    def to_add_spectral_data_from_sentinel2(self, band_names, band_keys, sentinel2_shot_metadata_address):
        # Чтение адресов изображений сниимка Sentinel-2 в заданных каналах (библиотека)
        band_table = to_get_spectral_data_from_sentinel2(band_names,
                                                         band_keys,
                                                         sentinel2_shot_metadata_address)
        # максимальное пространственное разрешение
        self.spec_space_res = min(list(band_table['spacial_resolution']))
        # список пространственных разрешений
        spec_space_res_list = []
        for band_key in band_keys:
            band_row = band_table[band_table['band_key'] == band_key]
            band_name = list(band_row['band_name'])[0]
            band_address = list(band_row['band_address'])[0]
            upper_wavelength = list(band_row['upper_wavelength'])[0]
            lower_wavelength = list(band_row['lower_wavelength'])[0]
            central_wavelength = list(band_row['central_wavelength'])[0]
            space_res = list(band_row['spacial_resolution'])[0]
            spec_space_res_list.append(space_res)
            # создание массивов спектральных данных по форме self.data_shape (объект Band)
            band = Band(band_name, band_address, space_res, upper_wavelength, lower_wavelength, central_wavelength,
                        self.data_shot_shape, rectangle_shape=self.rectangular_data_shot_shape,
                        multiplier=int(space_res / self.spec_space_res))
            self.spectral_data_dict.update({band_key: band})
        self.spec_geo_trans = self.spectral_data_dict[band_keys[0]].geo_trans
        self.spec_projection_ref = self.spectral_data_dict[band_keys[0]].projection_ref
        self.data_shape = self.spectral_data_dict[band_keys[0]].band.shape

    def to_add_spectral_data_from_worldview2(self, band_names, band_keys, shot_metadata_address):
        # Чтение адресов изображений сниимка Sentinel-2 в заданных каналах (библиотека)
        band_table = to_get_spectral_data_from_worldview2(band_names,
                                                         band_keys,
                                                         shot_metadata_address)
        # максимальное пространственное разрешение
        self.spec_space_res = min(list(band_table['spacial_resolution']))
        # список пространственных разрешений
        spec_space_res_list = []
        for band_key in band_keys:
            band_row = band_table[band_table['band_key'] == band_key]
            band_name = list(band_row['band_name'])[0]
            band_layer = list(band_row['band_layer'])[0]
            band_address = list(band_row['band_address'])[0]
            upper_wavelength = list(band_row['upper_wavelength'])[0]
            lower_wavelength = list(band_row['lower_wavelength'])[0]
            central_wavelenght = list(band_row['central_wavelength'])[0]
            space_res = list(band_row['spacial_resolution'])[0]
            calibration_coef = list(band_row['calibration_coef'])[0]

            spec_space_res_list.append(space_res)
            # создание массивов спектральных данных по форме self.data_shape (объект Band)
            band = Band(band_name, band_address, space_res, upper_wavelength, lower_wavelength, central_wavelenght,
                        self.data_shot_shape, band_layer=band_layer, calibration_coef=calibration_coef,
                        rectangle_shape=self.rectangular_data_shot_shape, multiplier=int(space_res / self.spec_space_res))
            self.spectral_data_dict.update({band_key: band})

        self.spec_geo_trans = self.spectral_data_dict[band_keys[0]].geo_trans
        self.spec_projection_ref = self.spectral_data_dict[band_keys[0]].projection_ref
        self.data_shape = self.spectral_data_dict[band_keys[0]].band.shape

    def to_add_spectral_data_from_images(self, band_names, band_keys, shot_metadata_addresses,
                                         layers, upper_wavelengths, lower_wavelengths, calibration_coefficient,
                                         calibration_add_coefficient):
        # Чтение адресов изображений сниимка Sentinel-2 в заданных каналах (библиотека)
        band_table = to_get_spectral_data_from_images(band_names,
                                                      band_keys,
                                                      shot_metadata_addresses, layers,
                                                      upper_wavelengths, lower_wavelengths, calibration_coefficient,
                                                      calibration_add_coefficient)
        # максимальное пространственное разрешение
        self.spec_space_res = min(list(band_table['spacial_resolution']))
        # список пространственных разрешений
        spec_space_res_list = []
        for i, band_key in enumerate(band_keys):
            band_row = band_table[band_table['band_key'] == band_key]
            band_name = list(band_row['band_name'])[0]
            band_address = list(band_row['band_address'])[0]
            band_layer = layers[i]
            upper_wavelength = list(band_row['upper_wavelength'])[0]
            lower_wavelength = list(band_row['lower_wavelength'])[0]
            central_wavelength = list(band_row['central_wavelength'])[0]
            space_res = list(band_row['spacial_resolution'])[0]
            spec_space_res_list.append(space_res)
            # создание массивов спектральных данных по форме self.data_shape (объект Band)
            band = Band(band_name, band_address, space_res, upper_wavelength, lower_wavelength, central_wavelength,
                        self.data_shot_shape, band_layer=band_layer, rectangle_shape=self.rectangular_data_shot_shape,
                        multiplier=int(space_res / self.spec_space_res))
            self.spectral_data_dict.update({band_key: band})
        self.spec_geo_trans = self.spectral_data_dict[band_keys[0]].geo_trans
        self.spec_projection_ref = self.spectral_data_dict[band_keys[0]].projection_ref
        self.data_shape = self.spectral_data_dict[band_keys[0]].band.shape

    def take_shadows(self, offset=1, keys_list=None):
        if keys_list is None:
            keys_list = self.spectral_data_dict.keys()

        all_bands = list(self.spectral_data_dict.values())
        mean_image = []
        for band in all_bands:
            mean_image.append(band.band)
        mean_image = np.mean(np.array(mean_image), axis=0)
        local_thresh = skimage.filters.threshold_local(mean_image, 35, offset=offset)
        mask = (mean_image > local_thresh)
        return mask

    def remove_shadows(self, offset=1, keys_list=None):
        mask = self.take_shadows(offset=offset, keys_list=keys_list)
        self.remove_by_mask(mask)

    def remove_by_mask(self, mask):
        for key in self.spectral_data_dict.keys():
            self.spectral_data_dict[key].band = mask * self.spectral_data_dict[key].band
        for i in range(len(self.texture_data_dict)):
            for key in self.texture_data_dict[i].keys():
                for j in range(len(self.texture_data_dict[i][key][0, 0])):
                    for k in range(len(self.texture_data_dict[i][key][0, 0, j])):
                        self.texture_data_dict[i][key][:, :, j, k] = mask * self.texture_data_dict[i][key][:, :, j, k]

    def to_clip_shot(self, data_shot_shape):
        # shape-файл, задающий форму снимка
        if type(data_shot_shape) == str:
            self.data_shot_shape = shapefile.Reader(data_shot_shape).shapes()[0]
        else:
            self.data_shot_shape = data_shot_shape
        for key in self.spectral_data_dict.keys():
            self.spectral_data_dict[key].band, new_spec_geo_trans = to_clip_shot(self.spectral_data_dict[key].band, self.data_shot_shape, self.spec_geo_trans,
                                                                                  self.spec_projection_ref, rectangle_shape=False)
            self.spec_geo_trans = new_spec_geo_trans
        for i in range(len(self.texture_data_dict)):
            for key in self.texture_data_dict[i].keys():
                mat = []
                for j in range(len(self.texture_data_dict[i][key][0, 0])):
                    for k in range(len(self.texture_data_dict[i][key][0, 0, j])):
                        new_im, new_texture_geo_trans = to_clip_shot(self.texture_data_dict[i][key][:, :, j, k], self.data_shot_shape, self.spec_geo_trans,
                                                                                  self.spec_projection_ref, rectangle_shape=False)
                        mat.append(new_im)
        self.texture_geo_trans = new_texture_geo_trans

    def to_make_rgb(self, red_band_key, green_band_key, blue_band_key,
                    red_band_add=0, green_band_add=0, blue_band_add=0,
                    red_band_mult=1, green_band_mult=1, blue_band_mult=1):
        red_band = self.spectral_data_dict.setdefault(red_band_key).band
        green_band = self.spectral_data_dict.setdefault(green_band_key).band
        blue_band = self.spectral_data_dict.setdefault(blue_band_key).band
        common_max = np.min(np.array([np.max(red_band), np.max(green_band), np.max(blue_band)]))
        common_min = np.max(np.array([np.min(red_band), np.min(green_band), np.min(blue_band)]))
        # поиск каналов для составления rgb снимка по заданным ключам
        red_band = red_band - common_min
        red_band = (red_band / common_max) * MAX_8_BIT * red_band_mult + red_band_add
        green_band = green_band - common_min
        green_band = (green_band / common_max) * MAX_8_BIT * green_band_mult + green_band_add
        blue_band = blue_band - common_min
        blue_band = (blue_band / common_max) * MAX_8_BIT * blue_band_mult + blue_band_add
        alfa_band = np.zeros(red_band.shape)
        alfa_band[np.where(red_band + green_band + blue_band != 0)] = 1 * MAX_8_BIT
        # запись rgb изображения в self.rgb
        self.rgb_image = np.uint8(np.dstack((red_band, green_band, blue_band, alfa_band)))

    def to_add_texture_data(self, texture_image_address,
                            texture_names=TEXTURE_NAMES_LIST,
                            directions=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
                            distance=1,
                            window_width=None,
                            grad_count=None,
                            texture_data_linked_to_spec=False,
                            to_save_clipped_image_like_high_res=False):
        texture_list = texture_names
        # Чтение изображения
        rast = gdal.Open(texture_image_address)
        digital_im = []
        for i in range(1, rast.RasterCount + 1):
            digital_im.append(rast.GetRasterBand(i).ReadAsArray())
        digital_im = np.array(digital_im)
        #digital_im = tiff.imread(texture_image_address)
        #if len(digital_im.shape) < 3:
        #    digital_im = np.array([digital_im])
        # Чтение привязки
        self.texture_geo_trans = rast.GetGeoTransform()
        self.texture_projection_ref = rast.GetProjectionRef()
        # сохранение направлений смежности
        self.texture_adjacency_directions = directions
        self.distances = distance
        self.texture_data_dict, digital_im, self.spec_geo_trans, self.texture_geo_trans = to_get_textures_from_image(
            np.asarray(digital_im), self.data_shot_shape, self.texture_geo_trans, self.texture_projection_ref,
            self.spec_geo_trans, self.spec_projection_ref, self.data_shape, texture_list, directions, distance,
            window_width=window_width, grad_count=grad_count, rectangle_shape=self.rectangular_data_shot_shape,
            texture_data_linked_to_spec=texture_data_linked_to_spec)
        if self.spec_projection_ref is None:
            self.spec_projection_ref = self.texture_projection_ref
        # сохранение вырезанного изображения в self.high_res_image
        if to_save_clipped_image_like_high_res:
            self.high_res_image = digital_im

        # # вырезание изображения по заданной форме снимка
        # digital_im, self.texture_geo_trans = to_clip_shot(np.asarray(digital_im),
        #                                                   self.data_shot_shape,
        #                                                   self.texture_geo_trans,
        #                                                   self.texture_projection_ref,
        #                                                   rectangle_shape=self.rectangular_data_shot_shape)
        # # пространственное разрешение текстурных данных
        # self.texture_space_res = abs(self.texture_geo_trans[1])
        # # вычисление координат пикселке спектральных данных через пиксели текстурных данных
        # # координаты центров пикселей спектральных данных в собственной системе координат
        # pix_of_spec_data = np.float64(np.array(list(product(range(0, self.data_shape[0]),
        #                                                     range(0, self.data_shape[1]))))) + 0.5
        # # географические координаты центров пикселей спектральных данных
        # geo_spec_data = pix2coord(pix_of_spec_data, self.spec_geo_trans, self.spec_projection_ref)
        # # координаты центров пикселей спектральных данных в системе координат текстурных данных
        # pix_of_spec_data_in_texture_data = coord2pix(geo_spec_data, self.texture_geo_trans, self.texture_projection_ref)
        # # массив координат центров пикселей спектральных данных в системе координат текстурных данных
        # pix_of_spec_data_in_texture_array = np.array(pix_of_spec_data_in_texture_data).reshape((self.data_shape[0], self.data_shape[1], 2))
        # # вычисление ширины и полуширины окна
        # window_width = round(self.spec_space_res / self.texture_space_res)
        # if window_width % 2 == 0:
        #     window_width -= 1
        # half_window_width = int((window_width - 1) / 2)
        #
        # # градуировка изображения по trans_grey_grad_count
        # graded_digital_im = grey_grad_lean_trans(digital_im, trans_grey_grad_count)
        #
        # # нулевой массив, в который будут записываться текстурные данные
        # texture_zero_array = np.zeros((self.data_shape[0], self.data_shape[1], 4))
        # # формирование библиотеки текстурных данных
        # for texture in texture_list:
        #     self.texture_data_dict.update({texture: copy.deepcopy(texture_zero_array)})
        # for i in range(0, texture_zero_array.shape[0]):
        #     for j in range(0, texture_zero_array.shape[1]):
        #         window_center_x = pix_of_spec_data_in_texture_array[i, j][0]
        #         window_center_y = pix_of_spec_data_in_texture_array[i, j][1]
        #         window = graded_digital_im[window_center_x - half_window_width: window_center_x + half_window_width,
        #                                    window_center_y - half_window_width: window_center_y + half_window_width]
        #         current_textures = to_calc_textures(window, directions, texture_list)
        #         for texture_name in texture_list:
        #             self.texture_data_dict[texture_name][i][j] = current_textures[texture_name]
        # # сохранение направлений смежности
        # self.texture_adjacency_directions = directions
        # сохранение вырезанного изображения в self.high_res_image
        # if to_save_clipped_image_like_high_res:
            # self.high_res_image = digital_im

            # I = np.repeat(np.swapaxes(np.array([np.arange(1, window_width + 1)]), 0, 1), window_width, axis=1)
    # J = np.repeat(np.array([np.arange(1, window_width + 1)]), window_width, axis=0)
    # texture_zero_array = np.zeros((graded_digital_im.shape[0] - window_width + 1,
    #                                graded_digital_im.shape[1] - window_width + 1))
    # for texture in texture_list:
    #     self.texture_data_dict.update({texture: copy.deepcopy(texture_zero_array)})
    # for i in range(0, texture_zero_array.shape[0]):
    #     for j in range(0, texture_zero_array.shape[1]):
    #         window = graded_digital_im[i: i + window_width, j: j + window_width]
    #         current_textures = my_to_calc_textures(window, I, J, texture_list)
    #         for texture_name in texture_list:
    #             self.texture_data_dict[texture_name][i][j] = current_textures[texture_name]
    # self.texture_space_res = abs(self.texture_geo_trans[1])
    # # сохранение вырезонного изображения в self.high_res_image
    # if to_save_clipped_image_like_high_res:
    #     self.high_res_image = digital_im
    # if to_save_clipped_graded_image_like_high_res:
    #     self.grad_high_res_image = graded_digital_im

    def mask_by_polygons(self, shape_address):
        # загрузка shape-файла с диска
        shape = shapefile.Reader(shape_address)
        # все полигоны из shape-файла
        polygons_list = shape.shapes()

        mask = to_make_mask(self.data_shape, polygons_list, self.spec_geo_trans, self.spec_projection_ref)
        self.remove_by_mask(mask)

    def get_prior_shot(self, color_map):
        prior_mat = np.zeros((self.data_shape[0], self.data_shape[1], 4))
        for class_key in color_map.keys():
            class_inx = np.where(self.prior_classes == bytes(class_key, 'utf-8'))
            prior_mat[class_inx] = colors.to_rgba(color_map[class_key])
        return prior_mat * 255

    def to_save_image_as_geotiff(self, image, geo_trans, projection_ref, file_name, file_directory, normed=True, byte=True):
        # адрес сохраняемой карты
        file_address = "".join([file_directory, '\\', file_name, '.tif'])
        # проверка количество каналов карты
        if normed:
            image = image / np.max(image) * 255
        if len(np.shape(image)) < 3:
            reshaped_image = np.array([image])
        else:
            # переформатирование карты
            reshaped_image = np.moveaxis(image, -1, 0)
        bands_quantity = np.shape(reshaped_image)[0]
        # создание файла
        driver = gdal.GetDriverByName('GTiff')
        if byte:
            gdal_type = gdal.GDT_Byte
        else:
            gdal_type = gdal.GDT_Float32
        dataset = driver.Create(file_address, len(image[0]), len(image[:, 0]), bands_quantity, gdal_type)
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

    def to_save_data_shot(self, save_name, save_directory):
        """
        # @Описание:
        #     Метод сохраняет объект self на диск в виде файла с заданным именем в заданной директории. Объект может быть
        #         загружен с помощью метода to_load_data.
        # :param save_name: название создаваемого файла сохранения (String)
        # :param save_directory: адрес директории, в которую сохраняется файл сохранения (String)
        # :return: файл сохранения, содержащий данный объект, который можно загрузить с помощью метода to_load_data.
        #     Название айла - save_name и он записывается в директории по адресу save_directory
        """
        with open("".join([save_directory, '\\', save_name, '.file']), "wb") as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    def to_combine_data_in_hypercube(self, spec_features=None,
                                           texture_features=None,
                                           texture_dict=None,
                                           sort=True):
        # сортировка ключей к спектральным и текстурным данным
        if spec_features is None:
            spec_keys = list(self.spectral_data_dict.keys())
        else:
            spec_keys = spec_features
        if sort:
            spec_keys = sorted(spec_keys)
        if texture_features is None:
            texture_keys = sorted(list(self.texture_data_dict.keys()))
        else:
            texture_keys = sorted(texture_features)
        # библиотека номеров направлений смежности
        if texture_dict is None:
            texture_adj_dir_dict = {0: {}}
            for texture in texture_keys:
                texture_adj_dir_dict[0][texture] = {}
            for texture in texture_keys:
                for dist in self.distances:
                    texture_adj_dir_dict[0][texture].update({dist: self.texture_adjacency_directions})
        else:
            texture_adj_dir_dict = texture_dict
        # формирование гиперкуба
        hypercube = []
        # добавление к гиперкубу спектральных данных из каналов по ключам из spec_keys в алфавитном порядке
        bands_list = []
        for spec_key in spec_keys:
            bands_list.append(self.spectral_data_dict[spec_key])
        im_list = []
        space_res_list = []
        geo_trans_list = []
        proj_ref_list = []
        for band in bands_list:
            im_list.append(band.band)
            space_res_list.append(band.space_res)
            geo_trans_list.append(band.geo_trans)
            proj_ref_list.append(band.projection_ref)
        # запись спектральных данных в гиперкуб
        for im in im_list:
            hypercube.append(im)
        # добавление к гиперкубу текстурных данные из каналов по ключам из texture_keys в алфавитном по направлениям
        # смежности, заданным texture_adj_dir_dict
        for layer in list(texture_adj_dir_dict.keys()):
            for texture_key in list(texture_adj_dir_dict[layer].keys()):
                for dist in texture_adj_dir_dict[layer][texture_key].keys():
                    directions = texture_adj_dir_dict[layer][texture_key][dist]
                    for texture_adj in directions:
                        dist_num = list(np.where(np.array(self.distances) == dist))[0][0]
                        texture_adj_num = list(np.where(np.array(self.texture_adjacency_directions) == texture_adj))[0][0]
                        hypercube.append(self.texture_data_dict[layer][texture_key][:, :, dist_num, texture_adj_num])
        return np.array(hypercube)

    def to_make_ndvi(self, red_band_key, nir_band_key):
        # формирование матрицы отражения
        spectral_reflection_hypercube = self.to_combine_data_in_hypercube([red_band_key, nir_band_key], texture_features=[], sort=False)
        # поиск каналов для составления ndvi снимка по заданным ключам
        red_band = np.int32(spectral_reflection_hypercube[0])
        nir_band = np.int32(spectral_reflection_hypercube[1])
        # вычисление NDVI
        np.seterr(divide='ignore', invalid='ignore')
        high_ndvi = nir_band + red_band
        ndvi = np.where(high_ndvi != 0, (nir_band - red_band) / high_ndvi, np.nan)
        self.ndvi = ndvi#np.round((ndvi + 1) * MAX_8_BIT / 2)

    def to_make_nbr(self, nir_band_key, swir_band_key):
        # формирование матрицы отражения
        spectral_reflection_hypercube = self.to_combine_data_in_hypercube([nir_band_key, swir_band_key], texture_features=[], sort=False)
        # поиск каналов для составления ndvi снимка по заданным ключам
        nir_band = np.int32(spectral_reflection_hypercube[0])
        swir_band = np.int32(spectral_reflection_hypercube[1])
        # вычисление NBR
        np.seterr(divide='ignore', invalid='ignore')
        high_ndvi = nir_band + swir_band
        nbr = np.where(high_ndvi != 0, (nir_band - swir_band) / high_ndvi, np.nan)
        self.nbr = nbr#np.round((nbr + 1) * MAX_8_BIT / 2)

    def to_make_ndre(self, red_edge_band_key, nir_band_key):
        # формирование матрицы отражения
        spectral_reflection_hypercube = self.to_combine_data_in_hypercube([red_edge_band_key, nir_band_key], texture_features=[], sort=False)
        # поиск каналов для составления ndvi снимка по заданным ключам
        red_edge_band = np.int32(spectral_reflection_hypercube[0])
        nir_band = np.int32(spectral_reflection_hypercube[1])
        # вычисление NDVI
        np.seterr(divide='ignore', invalid='ignore')
        high_ndre = nir_band + red_edge_band
        ndre = np.where(high_ndre != 0, (nir_band - red_edge_band) / high_ndre, np.nan)
        self.ndre = ndre#np.round((ndvi + 1) * MAX_8_BIT / 2)

    def to_make_pseudo_color_image(self, image_name, image_address, red_band_key, green_band_key, blue_band_key):
        # формирование матрицы
        spectral_reflection_hypercube = self.to_combine_data_in_hypercube([red_band_key, green_band_key, blue_band_key], texture_features=[])
        spectral_reflection_hypercube = np.swapaxes(np.swapaxes(spectral_reflection_hypercube, 0, 2), 0, 1)
        self.to_save_image_as_geotiff(spectral_reflection_hypercube, self.spec_geo_trans, self.spec_projection_ref, image_name,
                                      image_address)

    def to_add_prior_classes(self, my_map, itemsize=10):
        if not hasattr(self, 'prior_classes'):
            self.prior_classes = np.chararray(self.data_shape, itemsize=itemsize)
            self.prior_classes[:] = ''
        indexes_product = np.array(list(product(np.arange(self.prior_classes.shape[0]), np.arange(self.prior_classes.shape[1]))))
        shot_pixel_coordinates = pix2coord(indexes_product, self.spec_geo_trans, self.spec_projection_ref)
        map_pixels = np.int32(np.array(coord2pix(shot_pixel_coordinates, my_map.geo_trans, my_map.projection_ref)))
        inxs = np.where((map_pixels[:, 0] < my_map.map.shape[0]) & (map_pixels[:, 1] < my_map.map.shape[1]))
        map_pixels = map_pixels[inxs]
        indexes_product = indexes_product[inxs]
        prior_colors_len = my_map.map[map_pixels[:, 0], map_pixels[:, 1]]
        prior_classes_len = np.chararray(prior_colors_len.shape[0], itemsize=itemsize)
        prior_classes_len[:] = ''
        for i, col in enumerate(my_map.color_map):
            prior_classes_len[np.where((prior_colors_len[:, 0] == col[0]) & (prior_colors_len[:, 1] == col[1]) &
                                       (prior_colors_len[:, 2] == col[2]) & (prior_colors_len[:, 3] == col[3]))] = \
                my_map.keys_list[i]
        for class_key in my_map.keys_list:
            class_key_inx = indexes_product[np.where(prior_classes_len == bytes(class_key, 'utf-8'))]
            self.prior_classes[class_key_inx[:, 0], class_key_inx[:, 1]] = class_key

    # def to_add_prior_classes(self, my_map, itemsize=10):
    #     if not hasattr(self, 'prior_classes'):
    #         self.prior_classes = np.array([([itemsize * ' '] * self.data_shape[1])] * self.data_shape[0])
    #     indexes_product = np.array(list(product(np.arange(self.prior_classes.shape[0]), np.arange(self.prior_classes.shape[1]))))
    #     shot_pixel_coordinates = pix2coord(indexes_product, self.spec_geo_trans, self.spec_projection_ref)
    #     map_pixels = np.int32(np.array(coord2pix(shot_pixel_coordinates, my_map.geo_trans, my_map.projection_ref)))
    #     inxs = np.where((map_pixels[:, 0] < my_map.map.shape[0]) & (map_pixels[:, 1] < my_map.map.shape[1]))
    #     map_pixels = map_pixels[inxs]
    #     indexes_product = indexes_product[inxs]
    #     prior_colors_len = my_map.map[map_pixels[:, 0], map_pixels[:, 1]]
    #     prior_classes_len = np.chararray(prior_colors_len.shape[0])
    #     prior_classes_len[:] = ''
    #     for i, col in enumerate(my_map.color_map):
    #         prior_classes_len[np.where((prior_colors_len[:, 0] == col[0]) & (prior_colors_len[:, 1] == col[1]) &
    #                                    (prior_colors_len[:, 2] == col[2]) & (prior_colors_len[:, 3] == col[3]))] = \
    #             my_map.keys_list[i]
    #     for class_key in my_map.keys_list:
    #         class_key_inx = indexes_product[np.where(prior_classes_len == class_key)]
    #         self.prior_classes[class_key_inx[:, 0], class_key_inx[:, 1]] = class_key

    def transform_to_other_shot(self, other_shot, samples_shape_path, spec_bands_keys=None, exclude_shadow=False, offset=1,
                                accurate_pol=True, class_field=None, reg_color='r', color_map=None, class_mean=False):
        if spec_bands_keys is None:
            bands_keys = []
            for band_key in self.spectral_data_dict.keys():
                bands_keys.append(band_key)
        else:
            bands_keys = sorted(spec_bands_keys)

        # загрузка shape-файла с диска
        shape = shapefile.Reader(samples_shape_path)
        # все полигоны из shape-файла
        polygons_list = shape.shapes()

        field_of_class_number = 0
        if class_field is not None:
            for i in range(0, len(shape.fields)):
                if shape.fields[i][0] == class_field:
                    field_of_class_number = i - 1
                    break

        if exclude_shadow:
            shadow_mask = self.take_shadows(offset, bands_keys)
            self.remove_by_mask(shadow_mask)
            other_shot.remove_by_mask(shadow_mask)

        self_hypercube = self.to_combine_data_in_hypercube(spec_features=bands_keys, texture_features=[])
        other_hypercube = other_shot.to_combine_data_in_hypercube(spec_features=bands_keys, texture_features=[])

        self_samples = []
        other_samples = []
        class_list = []

        mask_shot_sum = np.sum(self.to_combine_data_in_hypercube(spec_features=spec_bands_keys, texture_features=[]), axis=0)
        mask_shot = np.where(mask_shot_sum == 0, 0, 1)
        mask_inx = np.where(mask_shot == 1)

        for k, pol in enumerate(polygons_list):
            self_clipped_image, self_new_geo_trans = to_clip_shot(self_hypercube,
                                                                  pol,
                                                                  self.spec_geo_trans,
                                                                  self.spec_projection_ref,
                                                                  accurate_pol=accurate_pol
                                                                  )
            other_clipped_image, other_new_geo_trans = to_clip_shot(other_hypercube,
                                                                    pol,
                                                                    other_shot.spec_geo_trans,
                                                                    other_shot.spec_projection_ref,
                                                                    accurate_pol=accurate_pol
                                                                    )

            mask_sum = np.sum(np.asarray(self_clipped_image), axis=0)
            mask_hypercube = np.where(mask_sum == 0, 0, 1)

            class_name = 'all'
            if class_field is not None:
                class_name = shape.record(k)[field_of_class_number]

            for i in range(len(mask_hypercube)):
                for j in range(len(mask_hypercube[0])):
                    if mask_hypercube[i, j] != 0:
                        self_samples.append(self_clipped_image[:, i, j])
                        other_samples.append(other_clipped_image[:, i, j])
                        class_list.append(class_name)
        self_samples = np.array(self_samples)
        other_samples = np.array(other_samples)
        class_list = np.array(class_list)
        prime_class_list = class_list
        class_set = sorted(list(set(class_list)))

        for i, band_key in enumerate(bands_keys):
            x = self_samples[:, i]
            y = other_samples[:, i]
            if class_mean:
                new_x = []
                new_y = []
                new_class = []
                for clas in class_set:
                    inxs = np.where(prime_class_list == clas)
                    new_x.append(np.mean(x[inxs]))
                    new_y.append(np.mean(y[inxs]))
                    new_class.append(clas)
                x = np.array(new_x)
                y = np.array(new_y)
                class_list = np.array(new_class)
            A = np.vstack([x, np.ones(len(x))]).T
            k, b = np.linalg.lstsq(A, y, rcond=None)[0]
            plt.figure(i)
            if class_field is not None:
                class_set = sorted(list(set(class_list)))
                if color_map is None:
                    color_map = {}
                    for clas in class_set:
                        color_map[clas] = None
                for clas in class_set:
                    inxs = np.where(class_list == clas)
                    plt.scatter(x[inxs], y[inxs], c=color_map[clas])
            else:
                plt.scatter(x, y, color_map)
            x1 = np.array([np.min(x), np.max(x)])
            plt.plot(x1, k * x1 + b, reg_color)
            plt.title(band_key)
            plt.xlabel(self.name)
            plt.ylabel(other_shot.name)
            plt.text(np.min(x), np.min(y), 'k = ' + str(k) + '; b = ' + str(b))
            if class_field is not None:
                plt.legend(['reg_line'] + class_set)

            self.spectral_data_dict[band_key].band[mask_inx] = k * self.spectral_data_dict[band_key].band[mask_inx] + b

    def graph_compare(self, other_shot, samples_shape_path, spec_bands_keys=None, exclude_shadow=False, offset=1,
                      accurate_pol=True, class_field=None, reg_color='r', color_map=None, class_mean=False, legend=True):
        if spec_bands_keys is None:
            bands_keys = []
            for band_key in self.spectral_data_dict.keys():
                bands_keys.append(band_key)
        else:
            bands_keys = sorted(spec_bands_keys)

        # загрузка shape-файла с диска
        shape = shapefile.Reader(samples_shape_path)
        # все полигоны из shape-файла
        polygons_list = shape.shapes()

        field_of_class_number = 0
        if class_field is not None:
            for i in range(0, len(shape.fields)):
                if shape.fields[i][0] == class_field:
                    field_of_class_number = i - 1
                    break

        if exclude_shadow:
            shadow_mask = self.take_shadows(offset, bands_keys)
            self.remove_by_mask(shadow_mask)
            other_shot.remove_by_mask(shadow_mask)

        self_hypercube = self.to_combine_data_in_hypercube(spec_features=bands_keys, texture_features=[])
        other_hypercube = other_shot.to_combine_data_in_hypercube(spec_features=bands_keys, texture_features=[])

        self_samples = []
        other_samples = []
        class_list = []

        mask_shot_sum = np.sum(self.to_combine_data_in_hypercube(spec_features=spec_bands_keys, texture_features=[]), axis=0)
        mask_shot = np.where(mask_shot_sum == 0, 0, 1)
        mask_inx = np.where(mask_shot == 1)

        for k, pol in enumerate(polygons_list):
            self_clipped_image, self_new_geo_trans = to_clip_shot(self_hypercube,
                                                                  pol,
                                                                  self.spec_geo_trans,
                                                                  self.spec_projection_ref,
                                                                  accurate_pol=accurate_pol
                                                                  )
            other_clipped_image, other_new_geo_trans = to_clip_shot(other_hypercube,
                                                                    pol,
                                                                    other_shot.spec_geo_trans,
                                                                    other_shot.spec_projection_ref,
                                                                    accurate_pol=accurate_pol
                                                                    )

            mask_sum = np.sum(np.asarray(self_clipped_image), axis=0)
            mask_hypercube = np.where(mask_sum == 0, 0, 1)

            class_name = 'all'
            if class_field is not None:
                class_name = shape.record(k)[field_of_class_number]

            for i in range(len(mask_hypercube)):
                for j in range(len(mask_hypercube[0])):
                    if mask_hypercube[i, j] != 0:
                        self_samples.append(self_clipped_image[:, i, j])
                        other_samples.append(other_clipped_image[:, i, j])
                        class_list.append(class_name)
        self_samples = np.array(self_samples)
        other_samples = np.array(other_samples)
        class_list = np.array(class_list)
        prime_class_list = class_list
        class_set = sorted(list(set(class_list)))

        for i, band_key in enumerate(bands_keys):
            x = self_samples[:, i]
            y = other_samples[:, i]
            if class_mean:
                new_x = []
                new_y = []
                new_class = []
                for clas in class_set:
                    inxs = np.where(prime_class_list == clas)
                    new_x.append(np.mean(x[inxs]))
                    new_y.append(np.mean(y[inxs]))
                    new_class.append(clas)
                x = np.array(new_x)
                y = np.array(new_y)
                class_list = np.array(new_class)
            A = np.vstack([x, np.ones(len(x))]).T
            #k, b = np.linalg.lstsq(A, y, rcond=None)[0]
            plt.figure(i)
            if class_field is not None:
                class_set = sorted(list(set(class_list)))
                if color_map is None:
                    color_map = {}
                    for clas in class_set:
                        color_map[clas] = None
                for clas in class_set:
                    inxs = np.where(class_list == clas)
                    plt.scatter(x[inxs], y[inxs], c=color_map[clas])
            else:
                plt.scatter(x, y, color_map)
            x1 = np.array([np.min(x), np.max(x)])
            #plt.plot(x1, k * x1 + b, reg_color)
            plt.title(band_key)
            plt.xlabel(self.name)
            plt.ylabel(other_shot.name)
            #plt.text(np.min(x), np.min(y), 'k = ' + str(k) + '; b = ' + str(b))
            if (class_field is not None) and legend:
                plt.legend(['reg_line'] + class_set)

            #self.spectral_data_dict[band_key].band[mask_inx] = k * self.spectral_data_dict[band_key].band[mask_inx] + b

    def graph_spectors(self, samples_shape_path, spec_bands_keys=None, exclude_shadow=False, offset=1,
                      accurate_pol=True, class_field=None, color_map=None, name=None, fig_num=1):
        if spec_bands_keys is None:
            bands_keys = []
            for band_key in self.spectral_data_dict.keys():
                bands_keys.append(band_key)
        else:
            bands_keys = sorted(spec_bands_keys)

        # загрузка shape-файла с диска
        shape = shapefile.Reader(samples_shape_path)
        # все полигоны из shape-файла
        polygons_list = shape.shapes()

        field_of_class_number = 0
        if class_field is not None:
            for i in range(0, len(shape.fields)):
                if shape.fields[i][0] == class_field:
                    field_of_class_number = i - 1
                    break

        if exclude_shadow:
            shadow_mask = self.take_shadows(offset, bands_keys)
            self.remove_by_mask(shadow_mask)

        self_hypercube = self.to_combine_data_in_hypercube(spec_features=bands_keys, texture_features=[])

        self_samples = []
        class_list = []

        mask_shot_sum = np.sum(self.to_combine_data_in_hypercube(spec_features=spec_bands_keys, texture_features=[]), axis=0)
        mask_shot = np.where(mask_shot_sum == 0, 0, 1)
        mask_inx = np.where(mask_shot == 1)

        for k, pol in enumerate(polygons_list):
            self_clipped_image, self_new_geo_trans = to_clip_shot(self_hypercube,
                                                                  pol,
                                                                  self.spec_geo_trans,
                                                                  self.spec_projection_ref,
                                                                  accurate_pol=accurate_pol
                                                                  )

            mask_sum = np.sum(np.asarray(self_clipped_image), axis=0)
            mask_hypercube = np.where(mask_sum == 0, 0, 1)

            class_name = 'all'
            if class_field is not None:
                class_name = shape.record(k)[field_of_class_number]

            for i in range(len(mask_hypercube)):
                for j in range(len(mask_hypercube[0])):
                    if mask_hypercube[i, j] != 0:
                        self_samples.append(self_clipped_image[:, i, j])
                        class_list.append(class_name)
        self_samples = np.array(self_samples)
        class_list = np.array(class_list)
        prime_class_list = class_list
        class_set = sorted(list(set(class_list)))

        x = []
        if color_map is None:
            color_map = {}
            for clas in class_set:
                color_map[clas] = None
        for band_key in bands_keys:
            x.append((self.spectral_data_dict[band_key].upper_wavelength +
                      self.spectral_data_dict[band_key].upper_wavelength) / 2)
        com_y = []
        for clas in class_set:
            y = self_samples
            inxs = np.where(prime_class_list == clas)
            com_y.append(np.mean(y[inxs], axis=0))
        com_y = np.array(com_y)
        zip_arr = sorted(zip(np.array(x), com_y.T), key=lambda t: t[0])
        x = np.array(zip_arr)[:, 0]
        com_y = []
        for i, wave_y in enumerate(zip_arr):
            com_y.append(wave_y[1])
        com_y = np.array(com_y).T
        plt.figure(fig_num)
        for i, clas in enumerate(class_set):
            plt.plot(x, com_y[i], color_map[clas], marker='o')
        plt.legend(class_set)
        if name is None:
            name = self.name
        plt.title(name)
        plt.xlabel('Wavelength')
        plt.grid()
        plt.show()

        return x, com_y

    def DOS_colebration(self, starting_band_key, model, mult=1, start_point='min',
                        show_correction_curve=False, show_start_point=False, map_band=None, neg_rule='none'):
        dark_object_value = None
        dark_object_pix_inx = None
        arr = self.spectral_data_dict[starting_band_key].band
        # preval = None
        # dark_object_value = 0
        # step = arr.max() / 255.0
        # for val in np.unique(arr):
        #     if val == 0:
        #         continue
        #     if preval is not None and (val - preval) < step:
        #         break
        #     else:
        #         preval = val
        # dark_object_value = preval
        if start_point == 'min':
            nonzero_inx = np.nonzero(arr)
            dark_object_value = np.min(arr[nonzero_inx])
            dark_object_arg = np.where(arr[nonzero_inx] == dark_object_value)
            dark_object_pix_inx = (nonzero_inx[0][dark_object_arg], nonzero_inx[1][dark_object_arg])
        elif start_point == isinstance(start_point, list) or isinstance(start_point, tuple):
            dark_object_pix_inx = start_point
            dark_object_value = arr[dark_object_pix_inx]
        elif start_point == isinstance(start_point, int) or start_point == isinstance(start_point, float):
            nonzero_inx = np.nonzero(arr)
            sorted_arr = np.sort(arr[nonzero_inx])
            dark_object_value = sorted_arr[np.int(len(sorted_arr) * start_point / 100)]
            dark_object_pix_inx = np.where(arr == dark_object_value)
        elif start_point == 'hist':
            nonzero_inx = np.nonzero(arr)
            plt.hist(arr[nonzero_inx], bins=np.arange(np.min(arr[nonzero_inx]), np.max(arr[nonzero_inx])))
            plt.show()
            dark_object_value = int(input('Enter dark object value: '))

        haze_coef_dict = {}
        starting_wavelength = (self.spectral_data_dict[starting_band_key].upper_wavelength +
                               self.spectral_data_dict[starting_band_key].upper_wavelength) / 2
        for key in self.spectral_data_dict.keys():
            band_wavelength = (self.spectral_data_dict[key].upper_wavelength +
                               self.spectral_data_dict[key].upper_wavelength) / 2
            haze_coef_dict[key] = ((band_wavelength ** model) / (starting_wavelength ** model)) * mult
        if neg_rule == 'none':
            for key in self.spectral_data_dict.keys():
                band = self.spectral_data_dict[key].band
                band = np.where(band > 0, band - dark_object_value * haze_coef_dict[key], 0)
                self.spectral_data_dict[key].band = band
        elif neg_rule == 'calib':
            min_colib_value_arr = []
            for key in self.spectral_data_dict.keys():
                nonzero_inx = np.nonzero(arr)
                band = self.spectral_data_dict[key].band
                min_colib_value_arr.append(np.min(band[nonzero_inx]) / haze_coef_dict[key])
            dark_object_value = np.min(np.array(min_colib_value_arr))
            for key in self.spectral_data_dict.keys():
                band = self.spectral_data_dict[key].band
                band = np.where(band > 0, band - dark_object_value * haze_coef_dict[key], 0)
                self.spectral_data_dict[key].band = band
        elif neg_rule == 'non-zero':
            for key in self.spectral_data_dict.keys():
                band = self.spectral_data_dict[key].band
                band = np.where(band > 0, band - dark_object_value * haze_coef_dict[key], 0)
                band[np.where(band < 0)] = 0
                self.spectral_data_dict[key].band = band
        wavelength_arr = []
        haze_arr = []
        for key in self.spectral_data_dict.keys():
            wavelength_arr.append((self.spectral_data_dict[key].upper_wavelength +
                                   self.spectral_data_dict[key].upper_wavelength) / 2)
            haze_arr.append(haze_coef_dict[key] * dark_object_value)
        fig_num = 0
        if show_correction_curve:
            plt.figure(fig_num)
            zip_arr = [wavelength_arr, haze_arr]
            zip_arr = sorted(zip_arr, key=lambda t: t[0])
            plt.plot(zip_arr[0], zip_arr[1])
            plt.show()
            fig_num += 1
        if map_band is None:
            map_band = start_point
        if show_start_point:
            plt.figure(fig_num)
            plt.imshow(self.spectral_data_dict[map_band].band)
            plt.scatter(dark_object_pix_inx[0], dark_object_pix_inx[1], c='red', marker='+')
            plt.show()

    def simple_DOS_colebration(self, start_point='min', show_correction_curve=False, map_band=None):
        dark_object_value_dict = {}
        for key in self.spectral_data_dict.keys():
            dark_object_value = None
            arr = self.spectral_data_dict[key].band
            if start_point == 'min':
                nonzero_inx = np.nonzero(arr)
                dark_object_value = np.min(arr[nonzero_inx])
                dark_object_arg = np.where(arr[nonzero_inx] == dark_object_value)
                dark_object_pix_inx = (nonzero_inx[0][dark_object_arg], nonzero_inx[1][dark_object_arg])
            elif start_point == isinstance(start_point, list) or isinstance(start_point, tuple):
                dark_object_pix_inx = start_point
                dark_object_value = arr[dark_object_pix_inx]
            elif start_point == isinstance(start_point, int) or start_point == isinstance(start_point, float):
                nonzero_inx = np.nonzero(arr)
                sorted_arr = np.sort(arr[nonzero_inx])
                dark_object_value = sorted_arr[np.int(len(sorted_arr) * start_point / 100)]
                dark_object_pix_inx = np.where(arr == dark_object_value)
            elif start_point == 'hist':
                nonzero_inx = np.nonzero(arr)
                plt.hist(arr[nonzero_inx], bins=np.arange(np.min(arr[nonzero_inx]), np.max(arr[nonzero_inx])))
                plt.title(key)
                plt.show()
                dark_object_value = int(input('Enter dark object value: '))
            dark_object_value_dict[key] = dark_object_value

        haze_coef_dict = {}

        for key in self.spectral_data_dict.keys():
            haze_coef_dict[key] = 1

        wavelength_arr = []
        haze_arr = []
        for key in self.spectral_data_dict.keys():
            wavelength_arr.append((self.spectral_data_dict[key].upper_wavelength +
                                   self.spectral_data_dict[key].upper_wavelength) / 2)
            haze_arr.append(haze_coef_dict[key] * dark_object_value_dict[key])
        for key in self.spectral_data_dict.keys():
            band = self.spectral_data_dict[key].band
            band = np.where(band > 0, band - dark_object_value_dict[key] * haze_coef_dict[key], 0)
            self.spectral_data_dict[key].band = band

        fig_num = 0
        if show_correction_curve:
            plt.figure(fig_num)
            zip_arr = [wavelength_arr, haze_arr]
            zip_arr = sorted(zip_arr, key=lambda t: t[0])
            plt.plot(zip_arr[0], zip_arr[1])
            plt.show()
            fig_num += 1
        if map_band is None:
            map_band = start_point
        #if show_start_point:
        #    plt.figure(fig_num)
        #    plt.imshow(self.spectral_data_dict[map_band].band)
        #    plt.scatter(dark_object_pix_inx[0], dark_object_pix_inx[1], c='red', marker='+')
        #    plt.show()

class Band:
    def __init__(self, band_name, band_address, space_res, upper_wavelength, lower_wavelength, central_wavelenght,
                 data_shot_shape, band_layer=None, calibration_coef=1, calibration_add=0, rectangle_shape=False, multiplier=1):
        # Название канала
        self.band_name = band_name
        # чтение данных из заданного изображения
        #try:
        #    image = np.array(Image.open(band_address))
        #except OSError:
        #    image = tiff.imread(band_address)
        rast = gdal.Open(band_address)
        image = rast.ReadAsArray()
        # Выбор слоя
        if band_layer is not None:
            image = image[band_layer]
        # Коллибровка изображения

        image = image * calibration_coef + calibration_add
        full_geo_trans = rast.GetGeoTransform()
        projection_ref = rast.GetProjectionRef()
        # перевод изображения в более высокое разрешение при необходимости
        if multiplier > 1:
            image = mult_res_increase(image, multiplier)
            full_geo_trans = (full_geo_trans[0],
                              full_geo_trans[1] / multiplier,
                              full_geo_trans[2],
                              full_geo_trans[3],
                              full_geo_trans[4],
                              full_geo_trans[5] / multiplier)
            # вырезание изображения по заданной форме снимка
        if data_shot_shape is not None:
            clipped_image, new_geo_trans = to_clip_shot(np.asarray(image), data_shot_shape, full_geo_trans,
                                                        projection_ref, rectangle_shape=rectangle_shape, accurate_pol=False)
        else:
            clipped_image = np.asarray(image)
            new_geo_trans = full_geo_trans
        self.band = clipped_image
        # данные геопривязки
        self.geo_trans = new_geo_trans
        self.projection_ref = projection_ref
        # пространственное разрешение спектральных данных
        self.space_res = space_res
        # верхняяи нижняя границы спектра канала (нм)
        self.upper_wavelength = upper_wavelength
        self.lower_wavelength = lower_wavelength
