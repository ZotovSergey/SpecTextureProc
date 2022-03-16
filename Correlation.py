import numpy as np
import pandas as pd
import shapefile
import gdal
import pickle
import tifffile as tiff
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D

# from DataShot import *
from GeoImageFunctions import *
from DataCollectionFunctions import to_get_textures_from_image

from Constants import TEXTURE_NAMES_LIST

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


class DataSet():
    def __init__(self):
        # Таблица, в которую будут собираться данные
        self.data_frame = pd.DataFrame()

    def to_collect_data_from_datashot(self, data_shot, shape_address, spec_features=None, texture_features=None,
                                      texture_adjacency_directions_dict=None, polygons_features=None, row_on_polygon=False):
        # сортировка ключей к спектральным и текстурным данным
        if spec_features is None:
            spec_keys = sorted(list(data_shot.spectral_data_dict.keys()))
        else:
            spec_keys = sorted(spec_features)
        if texture_features is None:
            texture_keys = sorted(list(data_shot.texture_data_dict.keys()))
        else:
            texture_keys = sorted(texture_features)
        # библиотека номеров направлений смежности
        if texture_adjacency_directions_dict is None:
            texture_adj_dir = data_shot.texture_adjacency_directions
            texture_adj_dir_dict = {}
            for texture_key in texture_keys:
                texture_adj_dir_dict.update({texture_key: texture_adj_dir})
        else:
            texture_adj_dir_dict = texture_adjacency_directions_dict
        # Чтение shape-файла
        shape = shapefile.Reader(shape_address)
        # Формирование гиперкуба
        hypercube = data_shot.to_combine_data_in_hypercube(spec_features=spec_keys,
                                                           texture_features=texture_keys,
                                                           texture_adjacency_directions_dict=
                                                           texture_adj_dir_dict)
        # Названия столбцов в таблице
        hypercube_tables_features_list = spec_keys
        for texture_key in texture_keys:
            for adj_dir in texture_adj_dir_dict[texture_key]:
                hypercube_tables_features_list.append(" ".join([texture_key, str(round(adj_dir / np.pi * 180))]))
        if polygons_features is None:
            polygons_tables_features_list = list(np.array(shape.fields)[:, 0][1:])
        else:
            polygons_tables_features_list = polygons_features
        # Образец пустой таблицы
        empty_table = {}

        for k, shot_feature in enumerate(hypercube_tables_features_list):
            empty_table.update({shot_feature: []})
        for k, pol_feature_name in enumerate(polygons_tables_features_list):
            empty_table.update({pol_feature_name: []})
        # Перебор полигонов и извлечение данных
        polygons = shape.shapes()
        for i, pol in enumerate(polygons):
            print(i)
            pol_table = copy.deepcopy(empty_table)
            new_hypercube = []
            for j in range(len(hypercube)):
                new_image, new_geo_trans = to_clip_shot(hypercube[j], pol, data_shot.spec_geo_trans,
                                                        data_shot.spec_projection_ref, rectangle_shape=False)
                new_hypercube.append(new_image)
            new_hypercube = np.array(new_hypercube)
            # Признаки из поилигонов
            pol_features = shape.records()[i]
            # Запись признаков в таблицу
            shot_feature=None
            for p in range(len(new_hypercube[0])):
                for q in range(len(new_hypercube[0, p])):
                    if np.any(new_hypercube[:, p, q] != 0):
                        for k, shot_feature in enumerate(hypercube_tables_features_list):
                            pol_table[shot_feature].append(new_hypercube[k, p, q])
            for k, pol_feature_name in enumerate(polygons_tables_features_list):
                pol_table[pol_feature_name].extend([pol_features[k]] * len(pol_table[shot_feature]))
            if row_on_polygon:
                for k, shot_feature in enumerate(hypercube_tables_features_list):
                    pol_table[shot_feature] = [np.array(pol_table[shot_feature]).mean()]
                # print(pol_table)
                for k, pol_feature_name in enumerate(polygons_tables_features_list):
                    pol_table[pol_feature_name] = [pol_table[pol_feature_name][0]]
            self.data_frame = self.data_frame.append(pd.DataFrame(pol_table))

    def to_balance_classes(self, class_ind_col):
        class_data_col = self.data_frame[class_ind_col].values
        classes = set(class_data_col)
        class_dict = {}
        for clas in classes:
            class_dict[clas] = np.where(class_data_col == clas)
        min_class_num = min([len(list(class_dict.values())[i][0]) for i in range(len(class_dict))])
        new_data_frame = pd.DataFrame()
        for clas in classes:
            shot_class_list = list(class_dict[clas][0])
            while len(shot_class_list) > min_class_num:
                del shot_class_list[np.random.randint(0, len(shot_class_list))]
            class_dict[clas] = shot_class_list
            new_data_frame = pd.concat([new_data_frame, self.data_frame.iloc[class_dict[clas]]])
        self.data_frame = new_data_frame

    def to_collect_data_from_image(self, image_address, shape_address, grad_count=None, texture_features=TEXTURE_NAMES_LIST,
                                   texture_adjacency_directions_dict=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4), distance=1, window_width=20,
                                   rectangle_shape=False, polygons_features=None, row_on_polygon=False, mark_dist=True,
                                   concat_type='index'):
        # чтение данных из заданного изображения
        image = np.array(tiff.imread(image_address))
        rast = gdal.Open(image_address)
        full_geo_trans = rast.GetGeoTransform()
        projection_ref = rast.GetProjectionRef()
        texture_keys = sorted(texture_features)
        # Чтение shape-файла
        shape = shapefile.Reader(shape_address)
        # Названия столбцов в таблице
        hypercube_tables_features_list = []
        for texture_key in texture_keys:
            for adj_dir in texture_adjacency_directions_dict:
                hypercube_tables_features_list.append(" ".join([texture_key, str(round(adj_dir / np.pi * 180))]))
        if mark_dist:
            new_hypercube_tables_features_list = []
            for feature in hypercube_tables_features_list:
                for dist in distance:
                    new_hypercube_tables_features_list.append(feature + ' dist_' + str(dist))
            hypercube_tables_features_list = new_hypercube_tables_features_list
        if polygons_features is None:
            polygons_tables_features_list = list(np.array(shape.fields)[:, 0][1:])
        else:
            polygons_tables_features_list = polygons_features
        # Образец пустой таблицы
        empty_table = {}

        for k, shot_feature in enumerate(hypercube_tables_features_list):
            empty_table.update({shot_feature: []})
        for k, pol_feature_name in enumerate(polygons_tables_features_list):
            empty_table.update({pol_feature_name: []})
        # Перебор полигонов и извлечение данных
        polygons = shape.shapes()
        for i, pol in enumerate(polygons):
            pol_table = copy.deepcopy(empty_table)
            # Приведение Изображений полигонов к единой яркости
            texture_data_dict, clipped_texture_im, new_spec_geo_trans, new_texture_geo_trans\
                = to_get_textures_from_image(image, pol, full_geo_trans, projection_ref,
                                             texture_list=texture_keys,
                                             texture_adjacency_directions=texture_adjacency_directions_dict,
                                             distance=distance, window_width=window_width, rectangle_shape=rectangle_shape,
                                             grad_count=grad_count)
            new_hypercube = []
            for texture_key in texture_data_dict.keys():
                new_hypercube.append(texture_data_dict[texture_key])
            new_hypercube = np.array(new_hypercube)
            # Признаки из поилигонов
            pol_features = shape.records()[i]
            # Запись признаков в таблицу
            for p in range(len(new_hypercube[0])):
                for q in range(len(new_hypercube[0, p])):
                    if np.any(new_hypercube[:, p, q] != 0):
                        for k in range(len(texture_keys)):
                            for b in range(len(new_hypercube[k, p, q])):
                                for d in range(len(new_hypercube[k, p, q, b])):
                                    shot_feature = hypercube_tables_features_list[k * len(new_hypercube[k, p, q]) * len(new_hypercube[k, p, q, b]) + b * len(new_hypercube[k, p, q, b]) + d]
                                    pol_table[shot_feature].append(new_hypercube[k, p, q, b, d])
                        for k, pol_feature_name in enumerate(polygons_tables_features_list):
                            pol_table[pol_feature_name].append(pol_features[k])
            if row_on_polygon:
                for k, shot_feature in enumerate(hypercube_tables_features_list):
                    pol_table[shot_feature] = [np.array(pol_table[shot_feature]).mean()]
                # print(pol_table)
                for k, pol_feature_name in enumerate(polygons_tables_features_list):
                    pol_table[pol_feature_name] = [pol_table[pol_feature_name][0]]

            self.data_frame = self.data_frame.append(pd.DataFrame(pol_table))

    def to_calc_pearson_correlation(self, x_features, y_feature, print_p_value=False):
        df_main = self.data_frame.dropna(subset=[y_feature])
        for x_feature in x_features:
            df = df_main.dropna(subset=[x_feature])
            corr_val = pearsonr(np.array(df[x_feature]), np.array(df[y_feature]))
            print_str = y_feature + '(' + x_feature + ')' + '   \t' + str(round(corr_val[0], 3))
            if print_p_value:
                print_str = print_str + ' p-value   \t' + str(round(corr_val[1], 3))
            print(print_str)

    def correlation_among_themselves(self, features, method='pearson', plot_graph=False, ticks_names=None):
        corr_table = self.data_frame[features].corr(method=method)
        if plot_graph:
            if ticks_names is None:
                ticks_names = np.array(corr_table.columns)
            plt.imshow(abs(corr_table.values), cmap=plt.get_cmap('jet'), norm=colors.Normalize(0.6, 1.0))
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=36)
            plt.title('Асимметрия', fontsize=60)
            plt.xlabel('d', fontsize=18)
            plt.ylabel('d', fontsize=18)
            plt.xticks(np.arange(len(corr_table.values)), ticks_names, rotation=0, fontsize=18)
            plt.yticks(np.arange(len(corr_table.values)), ticks_names, rotation=0, fontsize=18)
            plt.show()
        return corr_table

    def to_make_correlation_graph(self, x_feature, y_feature):
        fig, ax = plt.subplots()
        x = np.array(self.data_frame[x_feature])
        y = np.array(self.data_frame[y_feature])
        plt.scatter(x, y)
        id = self.data_frame['id'].values
        for i in range(len(x)):
            ax.annotate(id[i], (x[i], y[i]))
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.show()

    def bootstrap_confidence_intervals(self, first_col, second_col, n_samples=1000, score=np.mean, alpha=0.05,
                                       return_is_intersection=False):
        first_data_col = self.data_frame[first_col].values
        second_data_col = self.data_frame[second_col].values
        first_scores = list(map(score, get_bootstrap_samples(first_data_col, n_samples)))
        second_scores = list(map(score, get_bootstrap_samples(second_data_col, n_samples)))
        first_interval = stat_intervals(first_scores, alpha)
        second_interval = stat_intervals(second_scores, alpha)
        if return_is_intersection:
            return intervals_intersection(first_interval, second_interval)
        else:
            return first_interval, second_interval

    def plot_corr(self, first_col, second_col, class_column=None, colors_dict=None):
        fig, ax = plt.subplots()
        first_data_col = self.data_frame[first_col].values
        second_data_col = self.data_frame[second_col].values
        if class_column is None:
            scatter = ax.scatter(first_data_col, second_data_col)
        else:
            class_data_col = self.data_frame[class_column].values
            classes = set(class_data_col)
            if colors_dict is None:
                colors_dict = {}
                for clas in classes:
                    colors_dict.update({clas: np.random.rand(3)})
            else:
                if type(list(colors_dict.values())[0]) is str:
                    for clas in classes:
                        colors_dict[clas] = colors.to_rgb(colors_dict[clas])
            colors_col = np.zeros((len(class_data_col), 3))
            legend_elements = []
            for clas in classes:
                colors_col[np.where(class_data_col == clas)] = colors_dict[clas]
                ax.scatter(first_data_col, second_data_col, c=colors_col)
                legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_dict[clas],
                                              label=clas, markersize=7.5))
            ax.legend(handles=legend_elements)
        plt.xlabel(first_col)
        plt.ylabel(second_col)
        plt.show()

    def p_value_corr(self, first_col, second_col):
        print()

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

def concat_columns(first_dataset, second_dataset, common_columns):
    second_dataset_col = np.array(second_dataset.data_frame.columns.values)
    common_columns_inx_list = []
    for com_col in common_columns:
        common_columns_inx_list.append(list(np.where(com_col == second_dataset_col))[0][0])
    second_dataset_col = np.delete(second_dataset_col, common_columns_inx_list)
    new_dataset = DataSet()
    new_dataset.data_frame = pd.concat([first_dataset.data_frame, second_dataset.data_frame[second_dataset_col]], axis='columns')
    return new_dataset

# bootstrap
def get_bootstrap_samples(data, n_samples):
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples

def stat_intervals(stat, alpha):
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries

def intervals_intersection(first_interval, second_interval):
    is_intersection = False
    if (first_interval[0] >= second_interval[0]) and (first_interval[0] <= second_interval[1]):
        is_intersection = True
    if (first_interval[1] >= second_interval[0]) and (first_interval[0] <= second_interval[1]):
        is_intersection = True
    if (second_interval[0] >= first_interval[0]) and (second_interval[0] <= first_interval[1]):
        is_intersection = True
    if (second_interval[1] >= first_interval[0]) and (second_interval[0] <= first_interval[1]):
        is_intersection = True
    return is_intersection