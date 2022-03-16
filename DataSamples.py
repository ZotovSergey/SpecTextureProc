import numpy as np
import matplotlib.colors as colors
import random
import tifffile as tiff
import shapefile
import gdal
import pickle
import os
from copy import deepcopy
from sklearn import model_selection
from matplotlib import pyplot as plt

from DataCollectionFunctions import *
from GeoImageFunctions import *


TEXTURE_NAMES_LIST = ['Autocorrelation', 'ClusterProminence', 'ClusterShade', 'Contrast', 'Correlation', 'DiffEntropy',
                      'DiffVariance', 'Dissimilarity', 'Energy', 'Entropy', 'Homogeneity', 'Homogeneity2',
                      'InfMeasureCorr1', 'InfMeasureCorr2', 'MaxProb', 'SumAverage', 'SumEntropy', 'SumSquares',
                      'SumVariance']


def to_load_samples_set(save_address):
    with open(save_address, "rb") as file:
        loaded_spectrums_set = pickle.load(file)
    return loaded_spectrums_set

def to_select_samples(samples_shape_address, name_of_field_of_separation, train_samples_address=None,
                      test_samples_address=None, test_size=None, train_size=None, random_state=None, shuffle=True,
                      stratify=False):
    # загрузка shape-файла с диска
    shape = shapefile.Reader(samples_shape_address)
    # все полигоны из shape-файла
    polygons_list = shape.shapes()
    # cоздание shape-файла с обучающей выборкой
    if train_samples_address is not None:
        # создание нового shape-файла, если его нет
        if os.path.exists(train_samples_address):
            train_shape = shapefile.Editor(train_samples_address)
        else:
            train_shape = shapefile.Writer(shapefile.POLYGON)
        train_shape.fields = shape.fields
    else:
        train_shape = None
    # cоздание shape-файла с тестовой выборкой
    if test_samples_address is not None:
        # создание нового shape-файла, если его нет
        if os.path.exists(test_samples_address):
            test_shape = shapefile.Editor(test_samples_address)
        else:
            test_shape = shapefile.Writer(shapefile.POLYGON)
        test_shape.fields = shape.fields
    else:
        test_shape = None
    # поиск поля, по которому будет разделение образцов
    shape_fields = shape.fields
    field_of_separation_number = 0
    for i in range(0, len(shape_fields)):
        if shape_fields[i][0] == name_of_field_of_separation:
            field_of_separation_number = i - 1
            break
    X = list(zip(polygons_list, shape.records()))
    y = [shape.records()[i][field_of_separation_number] for i in range(len(shape.records()))]
    # разделение на обучающую и тестовую выборки
    if stratify:
        stratify_y = y
    else:
        stratify_y = None
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size,
                                                                        train_size=train_size,
                                                                        random_state=random_state,
                                                                        shuffle=shuffle, stratify=stratify_y)
    # запись выборок в shape-файлы и сохранение
    if train_shape is not None:
        train_shape._shapes.extend(np.array(X_train)[:, 0])
        train_shape.records.extend(np.array(X_train)[:, 1])
        train_shape.save(train_samples_address)
    if test_shape is not None:
        test_shape._shapes.extend(np.array(X_test)[:, 0])
        test_shape.records.extend(np.array(X_test)[:, 1])
        test_shape.save(test_samples_address)
    print()

class ClassSamplesSet:
    def __init__(self, samples_shape_address, spec_data_shot_address, texture_data_image_address,
                 name_of_field_of_separation, band_names, band_keys, layer=None, sat_name='Sentinel_2', texture_list=TEXTURE_NAMES_LIST,
                 texture_adjacency_directions=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4), distance=1,
                 window_width=None, grad_count=None, texture_data_linked_to_spec=False, average=False, maps_border=None,
                 rectangular_map=True, samples_type='classifier',
                 exclude_shadow=False, offset=5, accurate_pol=True, text_accurate_pol=False, shot=None):
        # библиотека образцов
        self.sample_dict = {}
        self.samples_type = samples_type
        # извлечение образцов
        if shot is None:
            self.to_add_samples(samples_shape_address, spec_data_shot_address, texture_data_image_address,
                                name_of_field_of_separation, band_names, band_keys, layer=layer, sat_name=sat_name,
                                texture_list=texture_list,
                                texture_adjacency_directions=texture_adjacency_directions, distance=distance,
                                window_width=window_width, grad_count=grad_count,
                                texture_data_linked_to_spec=texture_data_linked_to_spec, average=average,
                                maps_border=maps_border, rectangular_map=rectangular_map,
                                exclude_shadow=exclude_shadow, offset=offset, accurate_pol=accurate_pol, text_accurate_pol=text_accurate_pol)
        else:
            self.to_add_samples_by_shot(shot, samples_shape_address, name_of_field_of_separation,
                                        spec_band_keys=band_keys, texture_list=texture_list,
                                        texture_adjacency_directions=texture_adjacency_directions, distance=distance,
                                        accurate_pol=accurate_pol)

    def to_add_samples(self, samples_shape_address, spec_data_shot_address, texture_data_image_address,
                       name_of_field_of_separation, band_names, band_keys, layer=None, sat_name='Sentinel_2',
                       texture_list=TEXTURE_NAMES_LIST,
                       texture_adjacency_directions=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4), distance=1,
                       window_width=None, grad_count=None, texture_data_linked_to_spec=False, average=False,
                       maps_border=None, rectangular_map=True,
                       exclude_shadow=False, offset=5, accurate_pol=True, text_accurate_pol=False):
        # загрузка shape-файла с диска
        shape = shapefile.Reader(samples_shape_address)
        # все полигоны из shape-файла
        polygons_list = shape.shapes()
        # поиск поля, по которому будет разделение образцов
        shape_fields = shape.fields
        field_of_separation_number = 0
        for i in range(0, len(shape_fields)):
            if shape_fields[i][0] == name_of_field_of_separation:
                field_of_separation_number = i - 1
                break

        # чтение спектральных изображений и данных о них
        spec_digital_image_dict = {}
        spec_geo_trans_dict = {}
        spec_projection_ref_dict = {}
        space_res_dict = {}
        if spec_data_shot_address is not None:
            band_table = None
            if sat_name == 'Sentinel_2':
                # Чтение адресов изображений сниимка Sentinel-2 в заданных каналах (библиотека)
                band_table = to_get_spectral_data_from_sentinel2(band_names,
                                                                 band_keys,
                                                                 spec_data_shot_address)
            if sat_name == 'Landsat_8':
                # Чтение адресов изображений сниимка Landsat 8 в заданных каналах (библиотека)
                band_table = to_get_spectral_data_from_landsat8(band_names,
                                                                band_keys,
                                                                spec_data_shot_address)
            if sat_name == 'WorldView_2':
                # Чтение адресов изображений сниимка WorldView-2 в заданных каналах (библиотека)
                band_table = to_get_spectral_data_from_worldview2(band_names,
                                                                  band_keys,
                                                                  spec_data_shot_address)
            if sat_name == 'images':
                # Чтение адресов каналов из картинок tiff
                band_table = to_get_spectral_data_from_images(band_names,
                                                              band_keys,
                                                              spec_data_shot_address, layers=layer)
            max_res = min(band_table['spacial_resolution'])
            for band_key in band_keys:
                band_row = band_table[band_table['band_key'] == band_key]
                band_address = list(band_row['band_address'])[0]
                space_res = list(band_row['spacial_resolution'])[0]
                band_layer = list(band_row['band_layer'])[0]
                calibration_coef = list(band_row['calibration_coef'])[0]
                #try:
                #    image = np.array(Image.open(band_address))
                #except OSError:
                #    image = tiff.imread(band_address)
                rast = gdal.Open(band_address)
                image = rast.ReadAsArray()
                geo_trans = rast.GetGeoTransform()
                if len(image.shape) > 2:
                    image = image[band_layer]
                image = image * calibration_coef
                # приведение к общему разрешению
                multiplier = int(space_res / max_res)
                if multiplier > 1:
                    image = mult_res_increase(image, multiplier)
                    geo_trans = (geo_trans[0],
                                 geo_trans[1] / multiplier,
                                 geo_trans[2],
                                 geo_trans[3],
                                 geo_trans[4],
                                 geo_trans[5] / multiplier)
                band_image = np.asarray(image)
                space_res_dict.update({band_key: space_res})
                spec_digital_image_dict.update({band_key: band_image})
                spec_geo_trans_dict.update({band_key: geo_trans})
                spec_projection_ref_dict.update({band_key: rast.GetProjectionRef()})

        # чтение изображения высокого разрешения
        texture_digital_image = None
        texture_geo_trans = None
        texture_projection_ref = None
        if texture_data_image_address is not None:
            rast = gdal.Open(texture_data_image_address)
            texture_digital_image = []
            for i in range(1, rast.RasterCount + 1):
                texture_digital_image.append(rast.GetRasterBand(i).ReadAsArray())
            texture_digital_image = np.array(texture_digital_image)
            if len(texture_digital_image.shape) < 3:
                texture_digital_image = np.array([texture_digital_image])
            # for i, im in enumerate(texture_digital_image):
            #    texture_digital_image[i] = to_transform_matrix_grey_grad(im, grad_count)
            texture_geo_trans = rast.GetGeoTransform()
            texture_projection_ref = rast.GetProjectionRef()
            #if type(maps_border) == str:
            #    maps_shape = shapefile.Reader(maps_border).shapes()[0]
            #else:
            #    maps_shape = maps_border
            #new_texture_digital_images = []
            #for i in range(len(texture_digital_image)):
            #    # вырезание изображения по заданной форме polygon
            #    new_texture_digital_image, texture_geo_trans = to_clip_shot(texture_digital_image[i], maps_shape, texture_geo_trans,
            #                                                               texture_projection_ref, rectangle_shape=rectangular_map,
            #                                                               grad_count=grad_count)
            #    new_texture_digital_images.append(new_texture_digital_image)
        if self.samples_type == 'classifier':
            if exclude_shadow:
                self.remove_shadows(offset, spec_digital_image_dict)

            group_dict = {}
            polygons_groups = []
            for i in range(0, len(polygons_list)):
                class_name = str(shape.record(i)[field_of_separation_number])
                if class_name not in group_dict:
                    group_dict.update({class_name: len(group_dict)})
                    polygons_groups.append([])
                polygons_groups[group_dict[class_name]].append(polygons_list[i])
            # существующие образцы
            existed_classes = self.sample_dict.keys()
            diff_dist = True
            if type(distance) != list:
                distance = [copy.deepcopy(distance)]
                diff_dist = True
            # запись образцов
            for group_name in group_dict:
                # добавление нового класса при необходимости
                if group_name not in existed_classes:
                    # выбор случайного цвета для класса
                    color = colors.to_hex((random.random(), random.random(), random.random()))
                    self.sample_dict.update({group_name: ClassSample(group_name, color)})
                self.sample_dict[group_name].to_add_samples(polygons_groups[group_dict[group_name]],
                                                            spec_digital_image_dict, spec_geo_trans_dict,
                                                            spec_projection_ref_dict, space_res_dict, band_keys,
                                                            texture_digital_image, texture_geo_trans,
                                                            texture_projection_ref, texture_list,
                                                            texture_adjacency_directions,
                                                            distance=distance, window_width=window_width, grad_count=grad_count,
                                                            texture_data_linked_to_spec=texture_data_linked_to_spec,
                                                            average=average, accurate_pol=accurate_pol, text_accurate_pol=False)
            self.features_dict = {}
            if spec_data_shot_address is not None:
                # сортировка ключей в алфавитном порядке
                band_keys_sorted = band_keys
                len_band_keys_sorted = len(band_keys_sorted)
                for i in range(len(band_keys_sorted)):
                    self.features_dict.update({band_keys_sorted[i]: i})
            else:
                len_band_keys_sorted = 0
            if texture_data_image_address is not None:
                texture_list_sorted = sorted(texture_list)
                num = len(self.features_dict)
                for t in range(len(texture_digital_image)):
                    layer_ind = ''
                    if len(texture_digital_image) > 0:
                        layer_ind = ' layer_' + str(t)
                    for i in range(len(texture_list_sorted)):
                        for j in range(len(distance)):
                            for k in range(len(texture_adjacency_directions)):
                                texture_name = "-".join([texture_list_sorted[i], str(180 * texture_adjacency_directions[k] / np.pi)])
                                if diff_dist:
                                    texture_name = texture_name + ' dist_' + str(distance[j])
                                texture_name += layer_ind
                                self.features_dict.update({texture_name: num})
                                num += 1
        else:
            polygons_values = []
            for i in range(0, len(polygons_list)):
                value = float(shape.record(i)[field_of_separation_number])
                polygons_values.append((polygons_list[i], value))
            if type(distance) != list:
                distance = [copy.deepcopy(distance)]
                diff_dist = True
            # запись образцов
            self.sample_dict.update({'regression': RegressionSample()})
            self.sample_dict['regression'].to_add_samples(polygons_values,
                                                          spec_digital_image_dict, spec_geo_trans_dict,
                                                          spec_projection_ref_dict, space_res_dict, band_keys,
                                                          texture_digital_image, texture_geo_trans,
                                                          texture_projection_ref, texture_list,
                                                          texture_adjacency_directions,
                                                          distance=distance, window_width=window_width,
                                                          grad_count=grad_count,
                                                          texture_data_linked_to_spec=texture_data_linked_to_spec,
                                                          average=average, accurate_pol=accurate_pol)
            self.features_dict = {}
            if spec_data_shot_address is not None:
                # сортировка ключей в алфавитном порядке
                band_keys_sorted = band_keys
                len_band_keys_sorted = len(band_keys_sorted)
                for i in range(len(band_keys_sorted)):
                    self.features_dict.update({band_keys_sorted[i]: i})
            else:
                len_band_keys_sorted = 0
            if texture_data_image_address is not None:
                texture_list_sorted = sorted(texture_list)
                num = 0
                for t in range(len(texture_digital_image)):
                    layer_ind = ''
                    if len(texture_digital_image) > 0:
                        layer_ind = ' layer_' + str(t)
                    for i in range(len(texture_list_sorted)):
                        for j in range(len(distance)):
                            for k in range(len(texture_adjacency_directions)):
                                texture_name = "-".join(
                                    [texture_list_sorted[i], str(180 * texture_adjacency_directions[k] / np.pi)])
                                if diff_dist:
                                    texture_name = texture_name + ' dist_' + str(distance[j])
                                texture_name += layer_ind
                                self.features_dict.update({texture_name: num})
                                num += 1

    def to_add_samples_by_shot(self, shot, samples_shape_address, name_of_field_of_separation,
                               spec_band_keys=None, texture_list=None,
                               texture_adjacency_directions=None, distance=None, accurate_pol=True):
        # загрузка shape-файла с диска
        shape = shapefile.Reader(samples_shape_address)
        # все полигоны из shape-файла
        polygons_list = shape.shapes()
        # поиск поля, по которому будет разделение образцов
        shape_fields = shape.fields
        field_of_separation_number = 0
        for i in range(0, len(shape_fields)):
            if shape_fields[i][0] == name_of_field_of_separation:
                field_of_separation_number = i - 1
                break

        group_dict = {}
        polygons_groups = []
        for i in range(0, len(polygons_list)):
            class_name = str(shape.record(i)[field_of_separation_number])
            if class_name not in group_dict:
                group_dict.update({class_name: len(group_dict)})
                polygons_groups.append([])
            polygons_groups[group_dict[class_name]].append(polygons_list[i])
        # существующие образцы
        existed_classes = self.sample_dict.keys()
        diff_dist = True
        if type(distance) != list:
            distance = [copy.deepcopy(distance)]
            diff_dist = True
        # запись образцов
        for group_name in group_dict:
            # добавление нового класса при необходимости
            if group_name not in existed_classes:
                # выбор случайного цвета для класса
                color = colors.to_hex((random.random(), random.random(), random.random()))
                self.sample_dict.update({group_name: ClassSample(group_name, color)})
            self.sample_dict[group_name].to_add_samples_by_shot(polygons_groups[group_dict[group_name]],
                                                                shot, spec_band_list=spec_band_keys,
                                                                texture_list=texture_list,
                                                                texture_adjacency_directions=texture_adjacency_directions,
                                                                distance=distance, accurate_pol=accurate_pol)
        self.features_dict = {}
        if spec_band_keys is not None:
            # сортировка ключей в алфавитном порядке
            band_keys_sorted = spec_band_keys
            for i in range(len(band_keys_sorted)):
                self.features_dict.update({band_keys_sorted[i]: i})

        if texture_list is not None:
            texture_list_sorted = sorted(texture_list)
            num = len(self.features_dict)
            for i in range(len(texture_list_sorted)):
                for j in range(len(distance)):
                    for k in range(len(texture_adjacency_directions)):
                        texture_name = "-".join([texture_list_sorted[i], str(180 * texture_adjacency_directions[k] / np.pi)])
                        if diff_dist:
                            texture_name = texture_name + ' dist_' + str(distance[j]) + ' layer_0'
                        self.features_dict.update({texture_name: num})
                        num += 1

    def remove_shadows(self, offset, spec_digital_image_dict):
        keys_list = list(spec_digital_image_dict.keys())

        mean_image = []
        for key in keys_list:
            mean_image.append(spec_digital_image_dict[key])
        mean_image = np.mean(np.array(mean_image), axis=0)
        local_thresh = skimage.filters.threshold_local(mean_image, 35, offset=offset)
        mask = (mean_image > local_thresh)
        for key in keys_list:
            spec_digital_image_dict[key] = mask * spec_digital_image_dict[key]

    def to_rename(self, new_names_dict):
        # замена старых названий (ключи new_names_dict) на новые (значения new_names_dict)
        for name in new_names_dict:
            classes_sample = self.sample_dict[name]
            new_name = new_names_dict[name]
            classes_sample.name = new_name
            del self.sample_dict[name]
            self.sample_dict.update({new_name: classes_sample})

    def to_recolor(self, colors_dict):
        # замена в классах (ключи colors_dict) цветов на новые (значения colors_dict)
        if self.samples_type == 'classifier':
            for name in colors_dict:
                self.sample_dict[name].color = colors_dict[name]
        else:
            self.sample_dict['regression'].color = colors_dict

    def to_select_features(self, features):
        selected_samples = deepcopy(self)
        return selected_samples

    def to_save_samples_set(self, save_name, save_directory):
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

    def to_align_classes(self):
        # Копирование объекта self
        # new_classes = copy.deepcopy(self)
        new_classes = self
        keys_list = list(new_classes.sample_dict.keys())
        # вычисление класса с минимальным количеством образцов
        min_class_num = len(new_classes.sample_dict[keys_list[0]].samples)
        for key in keys_list[1:]:
            current_class_num = len(new_classes.sample_dict[key].samples)
            if min_class_num > current_class_num:
                min_class_num = current_class_num
        # исключение случайных образцов для каждого класса до тех пор, пока количество образцов не будет одинаковым и
        #   равным min_class_num
        for key in keys_list[1:]:
            current_class = new_classes.sample_dict[key]
            while (len(current_class.samples) > min_class_num):
                del current_class.samples[random.randint(0, len(current_class.samples) - 1)]
            new_classes.sample_dict.update({key: current_class})
        return new_classes

    def mean_spector(self, channels=None, classes=None, graph_colors=None):
        if channels is None:
            channels = self.features_dict.keys()
        channel_nums = []
        for channel in channels:
            channel_nums.append(self.features_dict[channel])
        if classes is None:
            classes = self.sample_dict.keys()
        if graph_colors is None:
            graph_colors = len(channels) * [None]
        graphs = []
        for clas in classes:
            graphs.append(np.mean(np.array(self.sample_dict[clas].samples)[:, channel_nums], axis=0))
        graphs = np.array(graphs)
        fig, ax = plt.subplots()
        for i in range(len(classes)):
            ax.plot(graphs[i], color=graph_colors[i])
        ax.set_xticks(range(len(channels)))
        ax.set_xticklabels(channels)
        ax.legend(classes)


class ClassSample:
    def __init__(self, name, color):
        self.name = name
        self.color = color
        self.samples = []

    def to_add_samples(self, polygons_list,
                       spectral_image_dict, spec_geo_trans_dict, spec_projection_ref_dict, space_res_dict, band_list,
                       texture_image, texture_geo_trans, texture_projection_ref, texture_list,
                       texture_adjacency_directions=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4), distance=1,
                       window_width=None, grad_count=None, texture_data_linked_to_spec=False, average=False,
                       accurate_pol=True, text_accurate_pol=False):
        common_spec_geo_trans = None
        spec_hypercube_projection_ref_dict = {None: None}
        band_keys = [None]
        polygon_shape = None
        # чтение данных для каждого полигона
        n = 0
        for polygon in polygons_list:
            n += 1
            print('id ' + str(n))
            print(str(polygon.points))
            hypercube = []
            if spectral_image_dict != {}:
                # сортировка ключей в алфавитном порядке
                band_keys = sorted(band_list)
                # чтение спектральных данных
                spec_hypercube_dict = deepcopy(spectral_image_dict)
                spec_hypercube_geo_trans_dict = deepcopy(spec_geo_trans_dict)
                spec_hypercube_projection_ref_dict = deepcopy(spec_projection_ref_dict)
                max_res = np.inf
                # for band_key in band_keys:
                #     clipped_image, new_geo_trans = to_clip_shot(spec_hypercube_dict[band_key],
                #                                                 polygon,
                #                                                 spec_hypercube_geo_trans_dict[band_key],
                #                                                 spec_hypercube_projection_ref_dict[band_key],
                #                                                 accurate_pol=accurate_pol
                #                                                 )
                #     spec_hypercube_dict[band_key] = clipped_image
                #     spec_hypercube_geo_trans_dict[band_key] = new_geo_trans
                #     if space_res_dict[band_key] < max_res:
                #         max_res = space_res_dict[band_key]
                #         common_spec_geo_trans = spec_hypercube_geo_trans_dict[band_key]
                # spec_hypercube = []
                # space_res_list = []
                # spec_geo_trans_list = []
                # spec_projection_ref_list = []
                # for band_key in band_keys:
                #     spec_hypercube.append(spec_hypercube_dict[band_key])
                #     space_res_list.append(space_res_dict[band_key])
                #     spec_geo_trans_list.append(spec_geo_trans_dict[band_key])
                #     spec_projection_ref_list.append(spec_projection_ref_dict[band_key])
                # polygon_shape = spec_hypercube[0].shape
                # # текстурных данных спектральных данных
                # hypercube = list(spec_hypercube)
                full_spec_hypercube = []
                for band_key in band_keys:
                    full_spec_hypercube.append(spec_hypercube_dict[band_key])
                full_spec_hypercube = np.array(full_spec_hypercube)
                if space_res_dict[band_key] < max_res:
                    max_res = space_res_dict[band_key]
                    common_spec_geo_trans = spec_hypercube_geo_trans_dict[band_key]
                clipped_image, new_geo_trans = to_clip_shot(full_spec_hypercube,
                                                            polygon,
                                                            common_spec_geo_trans,
                                                            spec_hypercube_projection_ref_dict[band_keys[0]],
                                                            accurate_pol=accurate_pol
                                                            )
                spec_hypercube = clipped_image
                common_spec_geo_trans = new_geo_trans
                space_res_list = []
                spec_geo_trans_list = []
                spec_projection_ref_list = []
                for band_key in band_keys:
                    space_res_list.append(space_res_dict[band_key])
                    spec_geo_trans_list.append(spec_geo_trans_dict[band_key])
                    spec_projection_ref_list.append(spec_projection_ref_dict[band_key])
                polygon_shape = spec_hypercube[0].shape
                # текстурных данных спектральных данных
                hypercube = list(spec_hypercube)
            if texture_image is not None:
                texture_list = sorted(texture_list)
                texture_data_dict = to_get_textures_from_image(texture_image, polygon, texture_geo_trans,
                                                               texture_projection_ref, common_spec_geo_trans,
                                                               spec_hypercube_projection_ref_dict[band_keys[0]], polygon_shape,
                                                               texture_list, texture_adjacency_directions, distance=distance,
                                                               window_width=window_width,
                                                               texture_data_linked_to_spec=texture_data_linked_to_spec,
                                                               rectangle_shape=False, accurate_pol=text_accurate_pol)[0]

                for k in range(len(texture_data_dict)):
                    for texture_key in texture_list:
                        for i in range(len(distance)):
                            for j in range(len(texture_adjacency_directions)):
                                hypercube.append(texture_data_dict[k][texture_key][:, :, i, j])
            hypercube = np.asarray(hypercube)

            if spectral_image_dict != {}:
                mask = np.sum(np.asarray(spec_hypercube), axis=0)
                mask = np.where(mask == 0, 0, 1)
                hypercube *= np.uint16(mask)

            new_samples = []
            for i in range(len(hypercube[0])):
                for j in range(len(hypercube[0][0])):
                    new_sample = []
                    for k in range(len(hypercube)):
                        new_sample.append(hypercube[k][i][j])
                    if new_sample[0] != 0:
                        new_samples.append(new_sample)
            if not average:
                self.samples.extend(new_samples)
            else:
                self.samples.append(list(np.mean(np.array(new_samples), axis=0)))

    def to_add_samples_by_shot(self, polygons_list,
                               shot, spec_band_list=None, texture_list=None,
                               texture_adjacency_directions=None, distance=None,
                               average=False, accurate_pol=True):
        if spec_band_list is None:
            spec_band_list = shot.spectral_data_dict.keys()
        if texture_list is None:
            texture_list = shot.texture_data_dict.keys()
        if texture_adjacency_directions is None:
            texture_adjacency_directions = shot.texture_adjacency_directions
        if distance is None:
            distance = shot.distances

        # чтение данных для каждого полигона
        n = 0

        texture_adj_dir_dict = {}
        for texture in texture_list:
            texture_adj_dir_dict[texture] = {}

        #for texture in texture_list:
        #    for dist in distance:
        #        texture_adj_dir_dict[texture].update({dist: texture_adjacency_directions})

        # чтение спектральных данных
        hypercube = shot.to_combine_data_in_hypercube(spec_band_list, texture_list)

        for polygon in polygons_list:
            n += 1
            print('id ' + str(n))
            clipped_image, new_geo_trans = to_clip_shot(hypercube,
                                                        polygon,
                                                        shot.spec_geo_trans,
                                                        shot.spec_projection_ref,
                                                        accurate_pol=accurate_pol
                                                        )

            new_samples = []
            for i in range(len(clipped_image[0])):
                for j in range(len(clipped_image[0][0])):
                    new_sample = []
                    for k in range(len(clipped_image)):
                        new_sample.append(clipped_image[k][i][j])
                    if new_sample[0] != 0:
                        new_samples.append(new_sample)
            if not average:
                self.samples.extend(new_samples)
            else:
                self.samples.append(list(np.mean(np.array(new_samples), axis=0)))


class RegressionSample:
    def __init__(self):
        self.samples = []
        self.color = None

    def to_add_samples(self, polygons_values,
                       spectral_image_dict, spec_geo_trans_dict, spec_projection_ref_dict, space_res_dict, band_list,
                       texture_image, texture_geo_trans, texture_projection_ref, texture_list,
                       texture_adjacency_directions=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4), distance=1,
                       window_width=None, grad_count=None, texture_data_linked_to_spec=False, average=False):
        common_spec_geo_trans = None
        spec_hypercube_projection_ref_dict = {None: None}
        band_keys = [None]
        polygon_shape = None
        # чтение данных для каждого полигона
        for polygon_value in polygons_values:
            hypercube = []
            if spectral_image_dict != {}:
                # сортировка ключей в алфавитном порядке
                band_keys = sorted(band_list)
                # чтение спектральных данных
                spec_hypercube_dict = deepcopy(spectral_image_dict)
                spec_hypercube_geo_trans_dict = deepcopy(spec_geo_trans_dict)
                spec_hypercube_projection_ref_dict = deepcopy(spec_projection_ref_dict)
                max_res = np.inf
                for band_key in band_keys:
                    clipped_image, new_geo_trans = to_clip_shot(spec_hypercube_dict[band_key],
                                                                polygon_value[0],
                                                                spec_hypercube_geo_trans_dict[band_key],
                                                                spec_hypercube_projection_ref_dict[band_key])
                    spec_hypercube_dict[band_key] = clipped_image
                    spec_hypercube_geo_trans_dict[band_key] = new_geo_trans
                    if space_res_dict[band_key] < max_res:
                        max_res = space_res_dict[band_key]
                        common_spec_geo_trans = spec_hypercube_geo_trans_dict[band_key]
                spec_hypercube = []
                space_res_list = []
                spec_geo_trans_list = []
                spec_projection_ref_list = []
                for band_key in band_keys:
                    spec_hypercube.append(spec_hypercube_dict[band_key])
                    space_res_list.append(space_res_dict[band_key])
                    spec_geo_trans_list.append(spec_geo_trans_dict[band_key])
                    spec_projection_ref_list.append(spec_projection_ref_dict[band_key])
                polygon_shape = spec_hypercube[0].shape
                # текстурных данных спектральных данных
                hypercube = list(spec_hypercube)
            if texture_image is not None:
                texture_list = sorted(texture_list)
                texture_data_dict = to_get_textures_from_image(texture_image, polygon_value[0], texture_geo_trans,
                                                               texture_projection_ref, common_spec_geo_trans,
                                                               spec_hypercube_projection_ref_dict[band_keys[0]], polygon_shape,
                                                               texture_list, texture_adjacency_directions, distance=distance,
                                                               window_width=window_width,
                                                               texture_data_linked_to_spec=texture_data_linked_to_spec,
                                                               rectangle_shape=False)[0]
                for k in range(len(texture_data_dict)):
                    for texture_key in texture_list:
                        for i in range(len(distance)):
                            for j in range(len(texture_adjacency_directions)):
                                hypercube.append(texture_data_dict[k][texture_key][:, :, i, j])
            hypercube = np.asarray(hypercube)
            new_samples = []
            for i in range(len(hypercube[0])):
                for j in range(len(hypercube[0][0])):
                    new_sample = []
                    for k in range(len(hypercube)):
                        new_sample.append(hypercube[k][i][j])
                    if new_sample[0] != 0:
                        new_samples.append((new_sample, polygon_value[1]))
            if not average:
                self.samples.extend(new_samples)
            else:
                self.samples.append(list(np.mean(np.array(new_samples), axis=0)))