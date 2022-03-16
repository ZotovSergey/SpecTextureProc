from numba.tests.test_svml import svml_funcs

import DataShot
import numpy as np
import math
import itertools
import statistics as sts
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import copy
import random
import gdal
import osr
import shapefile
import os.path
import copy
from shapely import geometry
from sklearn.linear_model import SGDClassifier, LogisticRegression, LinearRegression, Lasso
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.preprocessing import StandardScaler

from DataCollectionFunctions import to_get_spectral_data_from_sentinel2, to_get_textures_from_image, coord2pix
from GeoImageFunctions import *

def SVM(C=1.0, degree=3, kernel='rbf', gamma='auto', shrinking=True, class_weight=None, decision_function_shape='ecoc'):
    if decision_function_shape == 'ecoc':
        svm = SVC(C=C, degree=degree, kernel=kernel, gamma=gamma, shrinking=shrinking, class_weight=class_weight)
        svm = OutputCodeClassifier(svm)
    else:
        svm = SVC(C=C, degree=degree, kernel=kernel, gamma=gamma, shrinking=shrinking, class_weight=class_weight,
                  decision_function_shape=decision_function_shape)
    return svm

class Map:
    def __init__(self, samples_set, features=None, uniform_samples=None, non_classified=False, non_classified_color='black',
                 classes_borders=None, non_classified_random_points_number=1000, normalize=False):
        # образцы, классов и информация о самих классов
        self.samples_set = samples_set
        if uniform_samples is not None:
            if uniform_samples == 'Undersampling':
                # вычисление класса с минимальным количеством образцов
                samples_num = min([len(self.samples_set.sample_dict[key].samples) for key in list(self.samples_set.sample_dict.keys())])
                # исключение случайных образцов для каждого класса до тех пор, пока количество образцов не будет одинаковым и
                #   равным min_class_num
            if uniform_samples == 'Oversampling':
                # вычисление класса с минимальным количеством образцов
                samples_num = max([len(self.samples_set.sample_dict[key].samples) for key in list(self.samples_set.sample_dict.keys())])
                # исключение случайных образцов для каждого класса до тех пор, пока количество образцов не будет одинаковым и
                #   равным min_class_num

            if type(uniform_samples) is int:
                samples_num = uniform_samples
            self.samples_set, self.sample_remainder = sampling(self.samples_set, samples_num)
        # тип классификации
        self.type = None
        # список ключей, сортированных по алфавиту к образцам для выборки. Номер ключа в листе будет номером класса,
        #   используемым при обработке
        self.keys_list = sorted(list(self.samples_set.sample_dict.keys()))
        # подготовка обучающей выборки
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        if features is not None:
            self.features_keys = features
        else:
            self.features_keys = self.samples_set.features_dict.keys()
        self.X_train, self.y_train, self.X_test, self.y_test = to_prepare_selection(
            self.samples_set, features_keys_list=self.features_keys,
            non_classified=non_classified,
            non_classified_random_points_number=non_classified_random_points_number,
            characteristics_borders_dict=classes_borders)
        # перевод обозначений цветов в RGB
        if samples_set.samples_type == 'classifier':
            self.color_map = []
            for key in self.keys_list:
                self.color_map.append(list(colors.to_rgba(self.samples_set.sample_dict[key].color)))
        # добавление нового класса
        self.non_classified = non_classified
        if non_classified:
            self.keys_list.append('non-classified')
            self.color_map.append(colors.to_rgb(non_classified_color))
        # обученная машина, с помощью которой делалась карта
        self.mashine = None
        self.test_mashine = None
        # карта, полученная в результате обработки
        self.map = None
        # количество пикселей
        self.class_pixels_number = []
        # площади классов в кв. м
        self.pixel_space = 0
        # минимальные степени уверенности для каждого класса
        self.min_classes_confident_scores = None
        # данные о привязке
        self.geo_trans = None
        self.projection_ref = None
        # точность KFold кросс валидации для self.mashine с поряком
        self.k_fold_accuracy = None
        # порядок KFold кросс-валидации
        self.k_fold_order = None
        # точность на обучающей выборке
        self.accuracy_on_train_selection = None
        # Нормировачные коэффициенты
        self.norm_coef = np.zeros((len(self.X_train[0]), 2))
        self.norm_coef[:, 1] = 1
        #if normalize:
        #    self.norm_coef[:, 0] = np.min(self.X_test, axis=0)
        #    self.norm_coef[:, 1] = np.max(self.X_test, axis=0)
        #for i in range(len(self.X_train[0])):
        #    self.X_train[:, i] -= self.norm_coef[i, 0]
        #    self.X_train[:, i] /= self.norm_coef[i, 1] - self.norm_coef[i, 0]
        #    self.X_test[:, i] -= self.norm_coef[i, 0]
        #    self.X_test[:, i] /= self.norm_coef[i, 1] - self.norm_coef[i, 0]
        self.scaler = StandardScaler(with_mean=False, with_std=False)
        if normalize:
            #self.norm_coef[:, 0] = np.float64(np.mean(self.X_test, axis=0))
            #self.norm_coef[:, 1] = np.float64(np.std(self.X_test, axis=0))
            self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        #for i in range(len(self.X_train[0])):
        #    self.X_train[:, i] = self.X_train[:, i] - self.norm_coef[i, 0]
        #    self.X_train[:, i] = self.X_train[:, i] / self.norm_coef[i, 1]
        #    self.X_test[:, i] = self.X_test[:, i] - self.norm_coef[i, 0]
        #    self.X_test[:, i] = self.X_test[:, i] / self.norm_coef[i, 1]

    def add_map(self, other_map):
        copy_inx = np.where(other_map.map[:, :, 3] != 0)
        self.map[copy_inx] = other_map.map[copy_inx]
        self.keys_list = self.keys_list + other_map.keys_list
        self.color_map = self.color_map + other_map.color_map
        keys_color_zip = np.array(sorted(zip(self.keys_list, self.color_map)))
        self.keys_list = list(keys_color_zip[:, 0])
        self.color_map = list(keys_color_zip[:, 1])
        for samp_key in other_map.samples_set.sample_dict.keys():
            self.samples_set.sample_dict[samp_key] = other_map.samples_set.sample_dict[samp_key]
        return self

    def paint_by_mask(self, mask, color=(0, 0, 0, 0)):
        if isinstance(color, str):
            color = colors.to_rgba(color)
        paint_inx = np.where(mask != 0)
        self.map[paint_inx] = np.array(color)
        return self

    def to_fit_by_SGDC(self, greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10,
                       loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,
                       max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None,
                       random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5,
                       early_stopping=False, validation_fraction=0.1, n_iter_no_change=5,
                       class_weight=None, warm_start=False, average=False):
        if greed_search:
            # обучение методом квадратичного дискременантного анализа с сеткой параметров
            sgdc = SGDClassifier()
            grid_param = {}

            if type(loss) == list:
                grid_param['loss'] = loss
            if type(penalty) == list:
                grid_param['penalty'] = penalty
            if type(alpha) == list:
                grid_param['alpha'] = alpha
            if type(l1_ratio) == list:
                grid_param['l1_ratio'] = l1_ratio
            if type(fit_intercept) == list:
                grid_param['fit_intercept'] = fit_intercept
            if type(max_iter) == list:
                grid_param['max_iter'] = max_iter
            if type(tol) == list:
                grid_param['tol'] = tol
            if type(shuffle) == list:
                grid_param['shuffle'] = shuffle
            if type(verbose) == list:
                grid_param['verbose'] = verbose
            if type(epsilon) == list:
                grid_param['epsilon'] = epsilon
            if type(n_jobs) == list:
                grid_param['n_jobs'] = n_jobs
            if type(learning_rate) == list:
                grid_param['learning_rate'] = learning_rate
            if type(eta0) == list:
                grid_param['eta0'] = eta0
            if type(power_t) == list:
                grid_param['power_t'] = power_t
            if type(early_stopping) == list:
                grid_param['early_stopping'] = early_stopping
            if type(validation_fraction) == list:
                grid_param['validation_fraction'] = validation_fraction
            if type(n_iter_no_change) == list:
                grid_param['n_iter_no_change'] = n_iter_no_change
            if type(class_weight) == list:
                grid_param['class_weight'] = class_weight
            if type(warm_start) == list:
                grid_param['warm_start'] = warm_start
            if type(average) == list:
                grid_param['average'] = average

            self.mashine = model_selection.GridSearchCV(sgdc, grid_param, scoring=search_scoring, cv=cv)
        else:
            # обучение методом квадратичного дискременантного анализа без сетки параметров
            sgdc = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
                                 fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, shuffle=shuffle,
                                 verbose=verbose, epsilon=epsilon, n_jobs=n_jobs, random_state=random_state,
                                 learning_rate=learning_rate, eta0=eta0, power_t=power_t,
                                 early_stopping=early_stopping, validation_fraction=validation_fraction,
                                 n_iter_no_change=n_iter_no_change, class_weight=class_weight,
                                 warm_start=warm_start, average=average)
            self.mashine = sgdc
        self.type = 'SGDC'
        # обучение
        self.mashine.fit(self.X_train, self.y_train)
        if greed_search:
            self.test_mashine = copy.deepcopy(self.mashine.best_estimator_)
            print(self.mashine.best_params_)
            print(self.mashine.best_score_)
            print(self.mashine.best_estimator_.coef_)
        else:
            print(self.mashine.coef_)
            self.test_mashine = copy.deepcopy(self.mashine)
        # кросс валидация и обучение метода
        cross_validation_accuracy = self.StratifiedKFoldAccuracyCalc(self.X_test, self.y_test,
                                                                     k_fold_order=k_fold_order)
        self.k_fold_accuracy = cross_validation_accuracy
        self.k_fold_order = k_fold_order
        self.accuracy_on_train_selection = metrics.accuracy_score(self.mashine.predict(self.X_train), self.y_train)

    def to_fit_dict_by_SGDC(self, main_prob, rest_prob=1, greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10,
                            loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,
                            max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None,
                            random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5,
                            early_stopping=False, validation_fraction=0.1, n_iter_no_change=5,
                            warm_start=False, average=False):
        class_weight_dict = {}
        for i, class_key in enumerate(self.keys_list):
            class_weight = {}
            for j in range(len(self.keys_list)):
                class_weight[j] = rest_prob
            class_weight[i] = main_prob
            class_weight_dict[class_key] = class_weight

        self.estimator_dict = {}
        for class_key in self.keys_list:
            self.to_fit_by_SGDC(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                                k_fold_order=k_fold_order, loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
                                fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, shuffle=shuffle,
                                verbose=verbose, epsilon=epsilon, n_jobs=n_jobs, random_state=random_state,
                                learning_rate=learning_rate, eta0=eta0, power_t=power_t, early_stopping=early_stopping,
                                validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change,
                                class_weight=class_weight_dict[class_key], warm_start=warm_start, average=average)
            self.estimator_dict[class_key] = self.mashine
        self.to_fit_by_SGDC(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                            k_fold_order=k_fold_order, loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
                            fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, shuffle=shuffle,
                            verbose=verbose, epsilon=epsilon, n_jobs=n_jobs, random_state=random_state,
                            learning_rate=learning_rate, eta0=eta0, power_t=power_t, early_stopping=early_stopping,
                            validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change,
                            warm_start=warm_start, average=average)

    def to_fit_by_QDA(self, greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10, priors=None,
                      reg_param=0.0):
        if greed_search:
            # обучение методом квадратичного дискременантного анализа с сеткой параметров
            qda = QuadraticDiscriminantAnalysis()
            grid_param = {}
            if type(priors) == list:
                grid_param['priors'] = priors
            if type(reg_param) == list:
                grid_param['reg_param'] = reg_param
            self.mashine = model_selection.GridSearchCV(qda, grid_param, scoring=search_scoring, cv=cv)
        else:
            # обучение методом квадратичного дискременантного анализа без сетки параметров
            if priors is not None:
                priors_list = []
                keys = sorted(priors.keys())
                for key in keys:
                    priors_list.append(priors[key])
            else:
                priors_list = None
            qda = QuadraticDiscriminantAnalysis(priors=priors_list, reg_param=reg_param)
            self.mashine = qda
        self.type = 'QDA'
        # обучение
        self.mashine.fit(self.X_train, self.y_train)
        if greed_search:
            self.test_mashine = copy.deepcopy(self.mashine.best_estimator_)
            print(self.mashine.best_params_)
            print(self.mashine.best_score_)
        else:
            self.test_mashine = copy.deepcopy(self.mashine)
        # кросс валидация и обучение метода
        cross_validation_accuracy = self.StratifiedKFoldAccuracyCalc(self.X_test, self.y_test,
                                                                     k_fold_order=k_fold_order)
        self.k_fold_accuracy = cross_validation_accuracy
        self.k_fold_order = k_fold_order
        self.accuracy_on_train_selection = metrics.accuracy_score(self.mashine.predict(self.X_train), self.y_train)
        return self.mashine

    def to_fit_dict_by_QDA(self, main_prob, rest_prob=1, greed_search=False, search_scoring='accuracy', cv=3,
                           k_fold_order=10, reg_param=0.0):
        class_weight_dict = {}
        for i, class_key in enumerate(self.keys_list):
            class_weight = {}
            for j in range(len(self.keys_list)):
                class_weight[j] = rest_prob
            class_weight[i] = main_prob
            class_weight_dict[class_key] = class_weight

        self.estimator_dict = {}
        for class_key in self.keys_list:
            self.to_fit_by_QDA(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                               k_fold_order=k_fold_order, priors=class_weight_dict[class_key], reg_param=reg_param)
            self.estimator_dict[class_key] = self.mashine
        self.to_fit_by_QDA(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                           k_fold_order=k_fold_order, reg_param=reg_param)

    def to_fit_by_SVM(self, greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10,
                      C=1.0, degree=3, kernel='rbf', gamma='auto', shrinking=True, class_weight=None,
                      decision_function_shape='ecoc'):
        if greed_search:
            # обучение методом SVM с сеткой параметров
            svm = SVC()

            grid_param = {}

            if type(C) == list:
                grid_param['C'] = C
            if type(degree) == list:
                grid_param['degree'] = degree
            if type(kernel) == list:
                grid_param['kernel'] = kernel
            if type(gamma) == list:
                grid_param['gamma'] = gamma
            if type(shrinking) == list:
                grid_param['shrinking'] = shrinking
            if type(class_weight) == list:
                grid_param['class_weight'] = class_weight
            if type(decision_function_shape) == list:
                grid_param['decision_function_shape'] = decision_function_shape

            self.mashine = model_selection.GridSearchCV(svm, grid_param, scoring=search_scoring, cv=cv)
        else:
            # обучение методом SVM без сетки параметров
            svm = SVC(C=C, degree=degree, kernel=kernel, gamma=gamma, shrinking=shrinking,class_weight=class_weight,
                      decision_function_shape=decision_function_shape)
            self.mashine = svm
        self.type = 'SVM'
        # обучение
        self.mashine.fit(self.X_train, self.y_train)
        if greed_search:
            self.test_mashine = copy.deepcopy(self.mashine.best_estimator_)
            print(self.mashine.best_params_)
            print(self.mashine.best_score_)
        else:
            self.test_mashine = copy.deepcopy(self.mashine)
        # кросс валидация и обучение метода
        if k_fold_order is None:
            self.k_fold_accuracy = None
            self.k_fold_order = None
        else:
            cross_validation_accuracy = self.StratifiedKFoldAccuracyCalc(self.X_test, self.y_test,
                                                                         k_fold_order=k_fold_order)
            self.k_fold_accuracy = cross_validation_accuracy
            self.k_fold_order = k_fold_order
        return self.mashine

    def to_fit_dict_by_SVM(self, main_prob, rest_prob=1, greed_search=False, search_scoring='accuracy', cv=3,
                           k_fold_order=10, C=1.0, degree=3, kernel='rbf', gamma='auto', shrinking=True,
                           decision_function_shape='ovo'):
        if isinstance(main_prob, int):
            class_weight_dict = {}
            for i, class_key in enumerate(self.keys_list):
                class_weight = {}
                for j in range(len(self.keys_list)):
                    class_weight[j] = rest_prob
                class_weight[i] = main_prob
                class_weight_dict[class_key] = class_weight
        else:
            class_weight_dict = main_prob
        self.maps_dict = {}
        for class_key in self.keys_list:
            prior_class_map = Map(self.samples_set, features=self.features_keys,
                                  non_classified=self.non_classified, normalize=hasattr(self, 'scaler'))
            prior_class_map.to_fit_by_SVM(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                                          k_fold_order=k_fold_order, C=C, degree=degree, kernel=kernel, gamma=gamma,
                                          shrinking=shrinking, decision_function_shape=decision_function_shape,
                                          class_weight=class_weight_dict[class_key])
            self.maps_dict[class_key] = prior_class_map
        self.to_fit_by_SVM(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                           k_fold_order=k_fold_order, C=C, degree=degree, kernel=kernel, gamma=gamma,
                           shrinking=shrinking, decision_function_shape=decision_function_shape)

    def to_fit_by_KNN(self, greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10, n_neighbors=50,
                      weights='uniform', algorithm='auto', p=2):#, class_weights=None):
        # обучение методом ближайших соседей с сеткой параметров
        if greed_search:
            knn = KNeighborsClassifier()

            grid_param = {}

            if type(n_neighbors) == list:
                grid_param['n_neighbors'] = n_neighbors
            if type(weights) == list:
                grid_param['weights'] = weights
            if type(algorithm) == list:
                grid_param['algorithm'] = algorithm
            if type(p) == list:
                grid_param['p'] = p

            self.mashine = model_selection.GridSearchCV(knn, grid_param, scoring=search_scoring, cv=cv)
        else:
            # обучение методом ближайших соседей без сетки параметров
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm, p=p, weights=weights)
            self.mashine = knn
        self.type = 'KNN'
        # обучение
        #if class_weights is None:
        self.mashine.fit(self.X_train, self.y_train)
        #else:
            # class_lens = []
            # for i, key in enumerate(self.keys_list):
            #     class_lens.append(int(class_weights[i] * len(np.where(self.y_train == i)[0])))
            # unbalanced_samples_set, reminder = sampling(self.samples_set, class_lens)
            # unbalanced_X_train, unbalanced_y_train, unbalanced_X_test, unbalanced_y_test = \
            #     to_prepare_selection(unbalanced_samples_set, features_keys_list=self.features_keys, samples_keys=list(self.samples_set.sample_dict.keys()))
            # unbalanced_X_train = self.scaler.transform(unbalanced_X_train)
            # self.mashine.fit(unbalanced_X_train, unbalanced_y_train)
        if greed_search:
            self.test_mashine = copy.deepcopy(self.mashine.best_estimator_)
            print(self.mashine.best_params_)
            print(self.mashine.best_score_)
        else:
            self.test_mashine = copy.deepcopy(self.mashine)
        # кросс валидация и обучение метода
        if k_fold_order is None:
            self.k_fold_accuracy = None
            self.k_fold_order = None
        else:
            cross_validation_accuracy = self.StratifiedKFoldAccuracyCalc(self.X_test, self.y_test,
                                                                         k_fold_order=k_fold_order)
            self.k_fold_accuracy = cross_validation_accuracy
            self.k_fold_order = k_fold_order
        return self.mashine

    def to_fit_dict_by_KNN(self, main_prob, rest_prob=1, greed_search=False, search_scoring='accuracy', cv=3,
                           k_fold_order=10, n_neighbors=50, weights='uniform', algorithm='auto', p=2):
        if isinstance(main_prob, int):
            class_weight_dict = {}
            for i, class_key in enumerate(self.keys_list):
                class_weight = {}
                for j in range(len(self.keys_list)):
                    class_weight[j] = rest_prob
                class_weight[i] = main_prob
                class_weight_dict[class_key] = class_weight
        else:
            class_weight_dict = main_prob
        self.maps_dict = {}
        for class_key in self.keys_list:
            class_lens = []
            for i, key in enumerate(self.keys_list):
                class_lens.append(int(class_weight_dict[class_key][i] * len(np.where(self.y_train == i)[0])))
            unbalanced_samples_set, reminder = sampling(self.samples_set, class_lens)
            prior_class_map = Map(unbalanced_samples_set, features=self.features_keys,
                                  non_classified=self.non_classified, normalize=hasattr(self, 'scaler'))
            prior_class_map.to_fit_by_KNN(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                                          k_fold_order=k_fold_order, n_neighbors=n_neighbors, weights=weights,
                                          algorithm='auto', p=p)
            self.maps_dict[class_key] = prior_class_map
        self.to_fit_by_KNN(greed_search=greed_search, search_scoring=search_scoring, cv=cv, k_fold_order=k_fold_order,
                           n_neighbors=n_neighbors, weights=weights, algorithm='auto', p=p)

    def to_fit_by_RF(self, greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10,
                     n_estimators=10, max_depth=3, class_weight='balanced'):
        if greed_search:
            # обучение методом квадратичного дискременантного анализа с сеткой параметров
            rf = RandomForestClassifier()

            grid_param = {}

            if type(n_estimators) == list:
                grid_param['n_estimators'] = n_estimators
            if type(max_depth) == list:
                grid_param['max_depth'] = max_depth
            if type(class_weight) == list:
                grid_param['class_weight'] = class_weight

            self.mashine = model_selection.GridSearchCV(rf, grid_param, scoring=search_scoring, cv=cv)
        else:
            # обучение методом квадратичного дискременантного анализа без сетки параметров
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight=class_weight)
            self.mashine = rf
        self.type = 'RF'
        # обучение
        self.mashine.fit(self.X_train, self.y_train)
        if greed_search:
            self.test_mashine = copy.deepcopy(self.mashine.best_estimator_)
            print(self.mashine.best_params_)
            print(self.mashine.best_score_)
        else:
            self.test_mashine = copy.deepcopy(self.mashine)
        # кросс валидация и обучение метода
        cross_validation_accuracy = self.StratifiedKFoldAccuracyCalc(self.X_test, self.y_test,
                                                                     k_fold_order=k_fold_order)
        self.k_fold_accuracy = cross_validation_accuracy
        self.k_fold_order = k_fold_order
        self.accuracy_on_train_selection = metrics.accuracy_score(self.mashine.predict(self.X_train), self.y_train)
        return self.mashine

    def to_fit_dict_by_RF(self, main_prob, rest_prob=1, greed_search=False, search_scoring='accuracy', cv=3,
                          k_fold_order=10, n_estimators=10, max_depth=3):
        class_weight_dict = {}
        for i, class_key in enumerate(self.keys_list):
            class_weight = {}
            for j in range(len(self.keys_list)):
                class_weight[j] = rest_prob
            class_weight[i] = main_prob
            class_weight_dict[class_key] = class_weight

        self.estimator_dict = {}
        for class_key in self.keys_list:
            self.to_fit_by_RF(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                              k_fold_order=k_fold_order, n_estimators=n_estimators, max_depth=max_depth,
                              class_weight=class_weight_dict[class_key])
            self.estimator_dict[class_key] = self.mashine
        self.to_fit_by_RF(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                          k_fold_order=k_fold_order, n_estimators=n_estimators, max_depth=max_depth)

    def to_fit_by_LR(self, greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10,
                     penalty=None, dual=None, tol=None, C=None, fit_intercept=None, intercept_scaling=None,
                     class_weight=None, random_state=None, solver=None, max_iter=None, multi_class=None, verbose=None,
                     warm_start=None, n_jobs=None):
        if penalty is None:
            penalty = ['l2']
        if dual is None:
            dual = [False]
        if tol is None:
            tol = [0.0001]
        if C is None:
            C = [1.0]
        if fit_intercept is None:
            fit_intercept = [True]
        if intercept_scaling is None:
            intercept_scaling = [1]
        if class_weight is None:
            class_weight = [None]
        if random_state is None:
            random_state = [None]
        if solver is None:
            solver = ['lbfgs']
        if max_iter is None:
            max_iter = [100]
        if multi_class is None:
            multi_class = ['auto']
        if verbose is None:
            verbose = [0]
        if warm_start is None:
            warm_start = [False]
        if n_jobs is None:
            n_jobs = [None]

        if greed_search:
            # обучение методом квадратичного дискременантного анализа с сеткой параметров
            lr = LogisticRegression()
            grid_param = {'penalty': penalty, 'dual': dual, 'tol': tol, 'C': C, 'fit_intercept': fit_intercept,
                          'intercept_scaling': intercept_scaling, 'class_weight': class_weight, 'solver': solver,
                          'max_iter': max_iter, 'multi_class': multi_class, 'verbose': verbose,
                          'warm_start': warm_start, 'n_jobs': n_jobs}
            self.mashine = model_selection.GridSearchCV(lr, grid_param, scoring=search_scoring, cv=cv)
        else:
            # обучение методом квадратичного дискременантного анализа без сетки параметров
            lr = LogisticRegression(penalty=penalty[0], dual=dual[0], tol=tol[0], C=C[0],
                                    fit_intercept=fit_intercept[0], intercept_scaling=intercept_scaling[0],
                                    class_weight=class_weight[0], random_state=random_state[0], solver=solver[0],
                                    max_iter=max_iter[0], multi_class=multi_class[0], verbose=verbose[0],
                                    warm_start=warm_start[0], n_jobs=n_jobs[0])
            self.mashine = lr
        self.type = 'LR'
        # обучение
        self.mashine.fit(self.X_train, self.y_train)
        if greed_search:
            self.test_mashine = copy.deepcopy(self.mashine.best_estimator_)
            print(self.mashine.best_params_)
            print(self.mashine.best_score_)
        else:
            self.test_mashine = copy.deepcopy(self.mashine)
        # кросс валидация и обучение метода
        cross_validation_accuracy = self.StratifiedKFoldAccuracyCalc(self.X_test, self.y_test,
                                                                     k_fold_order=k_fold_order)
        self.k_fold_accuracy = cross_validation_accuracy
        self.k_fold_order = k_fold_order
        self.accuracy_on_train_selection = metrics.accuracy_score(self.mashine.predict(self.X_train), self.y_train)
        return self.mashine

    def to_fit_by_LR_reg(self, greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10,
                         fit_intercept=None, normalize=None, copy_X=None, n_jobs=None, positive=None):
        if fit_intercept is None:
            fit_intercept = [True]
        if normalize is None:
            normalize = [False]
        if copy_X is None:
            copy_X = [True]
        if n_jobs is None:
            n_jobs = [None]
        if positive is None:
            positive = [True]

        if greed_search:
            # обучение методом квадратичного дискременантного анализа с сеткой параметров
            lr_reg = LinearRegression()
            grid_param = {'fit_intercept': fit_intercept, 'normalize': normalize, 'copy_X': copy_X, 'n_jobs': n_jobs,
                          'positive': positive}
            self.mashine = model_selection.GridSearchCV(lr_reg, grid_param, scoring=search_scoring, cv=cv)
        else:
            # обучение методом квадратичного дискременантного анализа без сетки параметров
            lr_reg = LinearRegression(fit_intercept=fit_intercept[0], normalize=normalize[0], copy_X=copy_X[0],
                                  n_jobs=n_jobs[0])
            self.mashine = lr_reg
        self.type = 'LR_reg'
        # обучение
        self.mashine.fit(self.X_train, self.y_train)
        if greed_search:
            self.test_mashine = copy.deepcopy(self.mashine.best_estimator_)
            print(self.mashine.best_params_)
            print(self.mashine.best_score_)
        else:
            self.test_mashine = copy.deepcopy(self.mashine)
        return self.mashine

    def to_fit_by_lasso(self, greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10,
                        alpha=None, fit_intercept=None, normalize=None, precompute=None, copy_X=None, max_iter=None,
                        tol=None, warm_start=None, positive=None, random_state=None, selection=None):
        if alpha is None:
            alpha = [1.0]
        if fit_intercept is None:
            fit_intercept = [True]
        if normalize is None:
            normalize = [False]
        if precompute is None:
            precompute = [False]
        if copy_X is None:
            copy_X = [True]
        if max_iter is None:
            max_iter = [1000]
        if tol is None:
            tol = [0.0001]
        if warm_start is None:
            warm_start = [False]
        if positive is None:
            positive = [False]
        if random_state is None:
            random_state = [None]
        if selection is None:
            selection = ['cyclic']

        if greed_search:
            # обучение методом квадратичного дискременантного анализа с сеткой параметров
            lasso = Lasso()
            grid_param = {'alpha': alpha, 'fit_intercept': fit_intercept, 'normalize': normalize,
                          'precompute': precompute, 'copy_X': copy_X, 'max_iter': max_iter, 'tol': tol,
                          'warm_start': warm_start, 'positive': positive, 'random_state': random_state,
                          'selection': selection}
            self.mashine = model_selection.GridSearchCV(lasso, grid_param, scoring=search_scoring, cv=cv)
        else:
            # обучение методом квадратичного дискременантного анализа без сетки параметров
            lasso = Lasso(alpha=alpha[0], fit_intercept=fit_intercept[0], normalize=normalize[0],
                          precompute=precompute[0], copy_X=copy_X[0], max_iter=max_iter[0], tol=tol[0],
                          warm_start=warm_start[0], positive=positive[0], random_state=random_state[0],
                          selection=selection[0])
            self.mashine = lasso
        self.type = 'lasso'
        # обучение
        self.mashine.fit(self.X_train, self.y_train)
        if greed_search:
            self.test_mashine = copy.deepcopy(self.mashine.best_estimator_)
            print(self.mashine.best_params_)
            print(self.mashine.best_score_)
        else:
            self.test_mashine = copy.deepcopy(self.mashine)
        return self.mashine

    def to_fit_by_GPR(self, greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10,
                      kernel=None, alpha=None, optimizer=None, n_restarts_optimizer=None, normalize_y=None,
                      copy_X_train=None, random_state=None):
        if kernel is None:
            kernel = [None]
        if alpha is None:
            alpha = [1e-10]
        if optimizer is None:
            optimizer = ['fmin_l_bfgs_b']
        if n_restarts_optimizer is None:
            n_restarts_optimizer = [0]
        if normalize_y is None:
            normalize_y = [False]
        if copy_X_train is None:
            copy_X_train = [True]
        if random_state is None:
            random_state = [None]

        if greed_search:
            # обучение методом квадратичного дискременантного анализа с сеткой параметров
            gpr = GaussianProcessRegressor()
            grid_param = {'kernel': kernel, 'alpha': alpha, 'optimizer': optimizer,
                          'n_restarts_optimizer': n_restarts_optimizer, 'normalize_y': normalize_y,
                          'copy_X_train': copy_X_train, 'random_state': random_state}
            self.mashine = model_selection.GridSearchCV(gpr, grid_param, scoring=search_scoring, cv=cv)
        else:
            # обучение методом квадратичного дискременантного анализа без сетки параметров
            gpr = GaussianProcessRegressor(kernel=kernel[0], alpha=alpha[0], optimizer=optimizer[0],
                                           n_restarts_optimizer=n_restarts_optimizer[0], normalize_y=normalize_y[0],
                                           copy_X_train=copy_X_train[0], random_state=random_state[0])
            self.mashine = gpr
        self.type = 'GPR'
        # обучение
        self.mashine.fit(self.X_train, self.y_train)
        if greed_search:
            self.test_mashine = copy.deepcopy(self.mashine.best_estimator_)
            print(self.mashine.best_params_)
            print(self.mashine.best_score_)
        else:
            self.test_mashine = copy.deepcopy(self.mashine)
        return self.mashine

    def StratifiedKFoldAccuracyCalc(self, X, y, k_fold_order=10):
        # кросс-валидация stratified k-fold
        skf = model_selection.StratifiedKFold(n_splits=k_fold_order, shuffle=False)
        splits = skf.split(X, y)
        sum_error = 0
        for train_indices, test_indices in splits:
            self.test_mashine.fit(X[train_indices], y[train_indices])
            cross_y = self.test_mashine.predict(X[test_indices])

            sum_error += metrics.mean_absolute_error(y[test_indices], cross_y)
        accuracy = 1 - sum_error / float(k_fold_order)
        return accuracy

    def to_process(self, shot, spec_features=None, texture_features=None, texture_adjacency_directions_dict=None,
                   borders=None, cmap_gradation=1000, unbalanced_classifier=False, prior_class=None):
        # классификация пикселей из гиперкуба
        #   переформирование гиперкуба для обработки
        hypercube = shot.to_combine_data_in_hypercube(spec_features,
                                                      texture_features,
                                                      texture_adjacency_directions_dict)
        reshaped_hypercube = np.dstack(np.float64(hypercube))

        #for i in range(len(self.norm_coef)):
        #    reshaped_hypercube[:, :, i][np.where(reshaped_hypercube[:, :, i] != 0)] -= self.norm_coef[i, 0]
        #    reshaped_hypercube[:, :, i][np.where(reshaped_hypercube[:, :, i] != 0)] /= self.norm_coef[i, 1]
        if prior_class is None:
            indexes_list = np.nonzero(np.sum(reshaped_hypercube, axis=2))
            X = self.scaler.transform(reshaped_hypercube[indexes_list])
            y = self.mashine.predict(X)
        else:
            indexes_list = np.where((np.sum(reshaped_hypercube, axis=2) != 0) &
                                         (shot.prior_classes == bytes(prior_class, 'utf-8')))
            X = self.scaler.transform(reshaped_hypercube[indexes_list])
            y = self.mashine.predict(X)
        if self.samples_set.samples_type == 'classifier':
            # воссоздание карты и вычисление площади каждого класса
            map = np.zeros((len(reshaped_hypercube[:, 0]), len(reshaped_hypercube[0]), 4))
            for class_number in range(0, len(self.keys_list)):
                indexes = np.where(y == class_number)[0]
                map[indexes_list[0][indexes], indexes_list[1][indexes]] = self.color_map[class_number]
                # вычисление площади классов
                self.class_pixels_number.append(len(indexes))
            self.map = map
            # сохранение привязки
            self.geo_trans = shot.spec_geo_trans
            self.projection_ref = shot.spec_projection_ref
            self.pixel_space = self.geo_trans[1] ** 2
            if unbalanced_classifier:
                for class_key in self.keys_list:
                    self.maps_dict[class_key].to_process(shot, spec_features=spec_features,
                                                         texture_features=texture_features,
                                                         texture_adjacency_directions_dict=texture_adjacency_directions_dict,
                                                         borders=borders, cmap_gradation=cmap_gradation,
                                                          prior_class=class_key)
                    indexes_list = np.where((np.sum(reshaped_hypercube, axis=2) != 0) &
                                            (shot.prior_classes == bytes(class_key, 'utf-8')))
                    self.map[indexes_list] = self.maps_dict[class_key].map[indexes_list]
        else:
            indexes = np.arange(len(y))
            if borders is not None:
                y = np.where(y < borders[0], borders[0], y)
                y = np.where(y > borders[1], borders[1], y)
            if self.samples_set.sample_dict['regression'].color is not None:
                boundaries = np.float64(np.array(self.samples_set.sample_dict['regression'].color)[:, 0])
                col = np.array(self.samples_set.sample_dict['regression'].color)[:, 1]
                cmap = colors.LinearSegmentedColormap.from_list('forest_map', list(zip(boundaries, col)),
                                                                N=cmap_gradation)
                map = np.zeros((len(reshaped_hypercube[:, 0]), len(reshaped_hypercube[0]), 3))
                map[indexes_list[0][indexes], indexes_list[1][indexes]] = cmap(y)[:, 0:3]
            else:
                map = np.zeros((len(reshaped_hypercube[:, 0]), len(reshaped_hypercube[0])))
                map[indexes_list[0][indexes], indexes_list[1][indexes]] = y
            self.map = map
            # сохранение привязки
            self.geo_trans = shot.spec_geo_trans
            self.projection_ref = shot.spec_projection_ref
            self.pixel_space = self.geo_trans[1] ** 2

    def to_process_mask(self, shot, mask_keys_list, spec_features=None, texture_features=None, texture_adjacency_directions_dict=None,
                        borders=None, cmap_gradation=1000):
        # классификация пикселей из гиперкуба
        #   переформирование гиперкуба для обработки
        hypercube = shot.to_combine_data_in_hypercube(spec_features,
                                                      texture_features,
                                                      texture_adjacency_directions_dict)
        reshaped_hypercube = np.dstack(np.float64(hypercube))

        #for i in range(len(self.norm_coef)):
        #    reshaped_hypercube[:, :, i][np.where(reshaped_hypercube[:, :, i] != 0)] -= self.norm_coef[i, 0]
        #    reshaped_hypercube[:, :, i][np.where(reshaped_hypercube[:, :, i] != 0)] /= self.norm_coef[i, 1]
        indexes_list = np.nonzero(np.sum(reshaped_hypercube, axis=2))
        X = self.scaler.transform(reshaped_hypercube[indexes_list])
        y = self.mashine.predict(X)
        mask = np.zeros((len(reshaped_hypercube[:, 0]), len(reshaped_hypercube[0])))
        map = np.zeros((len(reshaped_hypercube[:, 0]), len(reshaped_hypercube[0]), 4))

        if self.samples_set.samples_type == 'classifier':
            # воссоздание карты и вычисление площади каждого класса
            for class_number, clas in enumerate(self.keys_list):
                class_number = self.keys_list.index(clas)
                indexes = np.where(y == class_number)[0]
                if clas in mask_keys_list:
                    mask[indexes_list[0][indexes], indexes_list[1][indexes]] = 1
                map[indexes_list[0][indexes], indexes_list[1][indexes]] = self.color_map[class_number]
                # вычисление площади классов
                self.class_pixels_number.append(len(indexes))
            self.map = map
            # сохранение привязки
            self.geo_trans = shot.spec_geo_trans
            self.projection_ref = shot.spec_projection_ref
            self.pixel_space = self.geo_trans[1] ** 2
        else:
            indexes = np.arange(len(y))
            if borders is not None:
                y = np.where(y < borders[0], borders[0], y)
                y = np.where(y > borders[1], borders[1], y)
            if self.samples_set.sample_dict['regression'].color is not None:
                boundaries = np.float64(np.array(self.samples_set.sample_dict['regression'].color)[:, 0])
                col = np.array(self.samples_set.sample_dict['regression'].color)[:, 1]
                cmap = colors.LinearSegmentedColormap.from_list('forest_map', list(zip(boundaries, col)),
                                                                N=cmap_gradation)
                map = np.zeros((len(reshaped_hypercube[:, 0]), len(reshaped_hypercube[0]), 3))
                map[indexes_list[0][indexes], indexes_list[1][indexes]] = cmap(y)[:, 0:3]
            else:
                map = np.zeros((len(reshaped_hypercube[:, 0]), len(reshaped_hypercube[0])))
                map[indexes_list[0][indexes], indexes_list[1][indexes]] = y
            self.map = map
            # сохранение привязки
            self.geo_trans = shot.spec_geo_trans
            self.projection_ref = shot.spec_projection_ref
            self.pixel_space = self.geo_trans[1] ** 2

        return mask

    def to_calc_dominant(self, areas_shp_address, dominant_shp_address, dominant_field_name, shot,
                         possible_dom_classes=None, fields_to_copy=None,
                         spec_features=None, texture_features=None, texture_adjacency_directions_dict=None,
                         average_before_pred=False, dom_order=1, min_space=0, note_field=None, space_field_name=None,
                         unbalanced_classifier=False):
        ## функция, заполняющая цветами колонку colors
        #def colors_column(table):
        #    colored_table = table.copy()
        #    colored_table[:] = ''
        #    for i in range(0, len(pos_dom_keys)):
        #        hex_color = colors.to_hex(self.color_map[i])
        #        colored_table['Colors'][i] = 'background-color: %s' % hex_color
        #    return colored_table
        if type(shot) == list:
            hypercube = []
            geo_trans = []
            projection_ref = []
            for one_shot in shot:
                hypercube.append(one_shot.to_combine_data_in_hypercube(spec_features=spec_features,
                                                                       texture_features=texture_features,
                                                                       texture_adjacency_directions_dict=texture_adjacency_directions_dict))
                geo_trans.append(one_shot.spec_geo_trans)
                projection_ref.append(one_shot.spec_projection_ref)
        else:
            hypercube = shot.to_combine_data_in_hypercube(spec_features=spec_features,
                                                          texture_features=texture_features,
                                                          texture_dict=texture_adjacency_directions_dict)
            geo_trans = shot.spec_geo_trans
            projection_ref = shot.spec_projection_ref
        pos_dom_num = []
        if possible_dom_classes is None:
            pos_dom_keys = self.keys_list
            pos_dom_num = range(0, len(self.keys_list))
        else:
            pos_dom_keys = possible_dom_classes
            keys_np = np.array(self.keys_list)
            for key in possible_dom_classes:
                pos_dom_num.append(np.where(keys_np == key)[0][0])
        # загрузка shape-файла с диска для чтения и редактирования
        initial_areas = shapefile.Reader(areas_shp_address)
        # создание нового shape-файла, если его нет
        if os.path.exists(dominant_shp_address):
            areas_with_dominant = shapefile.Editor(dominant_shp_address)
        else:
            areas_with_dominant = shapefile.Writer(shapefile.POLYGON)
            # копирование содержимого в новый файл
        areas_with_dominant._shapes.extend(initial_areas.shapes())
        if fields_to_copy is None:
            new_records = initial_areas.records()
            areas_with_dominant.fields = list(initial_areas.fields)
        else:
            fields_to_copy_numbers = []
            fields_to_dominants_areas = [initial_areas.fields[0]]
            initial_areas_fields = np.array(initial_areas.fields)[:, 0][1:]
            for i, field in enumerate(initial_areas_fields):
                if field in fields_to_copy:
                    fields_to_copy_numbers.append(i)
                    fields_to_dominants_areas.append(initial_areas.fields[i + 1])
            new_records = []
            for i in fields_to_copy_numbers:
                new_records.append([row[i] for row in initial_areas.records()])
            new_records = list(map(list, zip(*new_records)))
            areas_with_dominant.fields = list(fields_to_dominants_areas)
        if space_field_name is not None:
            fields_names_list = np.array(areas_with_dominant.fields[1:])
            if space_field_name not in fields_names_list:
                areas_with_dominant.field(space_field_name)
                [pol.append('') for pol in new_records]
                space_field_inx = np.array(new_records).shape[1] - 1
            else:
                space_field_inx = np.where(fields_names_list == space_field_name)[0][0]
        # добавление поля для доминантного класса, если его еще нет
        fields_names_list = np.array(areas_with_dominant.fields[1:])
        if dominant_field_name not in fields_names_list:
            areas_with_dominant.field(dominant_field_name)
            [pol.append('') for pol in new_records]
            dominant_field_inx = np.array(new_records).shape[1] - 1
        else:
            dominant_field_inx = np.where(fields_names_list == dominant_field_name)[0][0]
        # добавление поля для примечания, если необходим
        note_field_inx = None
        if note_field is not None:
            fields_names_list = np.array(areas_with_dominant.fields[1:])
            if note_field not in fields_names_list:
                areas_with_dominant.field(note_field)
                [pol.append('') for pol in new_records]
                note_field_inx = np.array(new_records).shape[1] - 1
            else:
                note_field_inx = np.where(fields_names_list == note_field)[0][0]
        # вычисление доминанты для каждого полигона
        pix_count = np.zeros(len(pos_dom_num))
        current_hypercube = hypercube
        current_geo_trans = geo_trans
        current_projection_ref = projection_ref
        for i, shp in enumerate(initial_areas.shapes()):
            if type(hypercube) == list:
                current_hypercube = hypercube[i]
                current_geo_trans = geo_trans[i]
                current_projection_ref = projection_ref[i]
            X_pol_list = []
            area_space = 0
            points_pix = np.array([list(i) for i in coord2pix(shp.points, current_geo_trans, current_projection_ref)])

            polygon_pix = geometry.Polygon(points_pix)
            x_pix_pol = np.int32(points_pix[:, 0])
            y_pix_pol = np.int32(points_pix[:, 1])
            up_pix = min(y_pix_pol)
            down_pix = max(y_pix_pol)
            left_pix = min(x_pix_pol)
            right_pix = max(x_pix_pol)
            pol_coord = []
            # отбор образцов из снимка, входящих в заданный полигон
            for y in range(up_pix, down_pix):
                for x in range(left_pix, right_pix):
                    if polygon_pix.contains(geometry.Point(x, y)):
                        area_space += 1
                        if any(current_hypercube[:, y, x] != 0):
                            X_pol_list.append(current_hypercube[:, y, x])
                            pol_coord.append([x, y])
            if len(X_pol_list) > 0:
                X_pol_list = self.scaler.transform(np.array(X_pol_list))
            if len(X_pol_list) > 0:
                if average_before_pred:
                    X_pol_list = np.array([np.mean(np.array(X_pol_list), axis=0)])
                y_pol_list = self.mashine.predict(X_pol_list)
                if unbalanced_classifier:
                    pol_coord = np.array(pol_coord)
                    for class_key in self.keys_list:
                        indexes_list = np.where(shot.prior_classes[np.array(pol_coord)[:, 1], np.array(pol_coord)[:, 0]] == bytes(class_key, 'utf-8'))
                        if len(indexes_list[0]) > 0:
                            y_pol_list[indexes_list] = self.maps_dict[class_key].mashine.predict(X_pol_list[indexes_list])
                dominant_class_inx = None
                max_count_classes_samples = 0
                count_classes_samples_list = []
                for j in pos_dom_num:
                    keys_samples_count = len(np.where(y_pol_list == j)[0])
                    count_classes_samples_list.append(keys_samples_count)
                    # if keys_samples_count > max_count_classes_samples:
                    #     max_count_classes_samples = keys_samples_count
                    #     dominant_class_inx = j
                if len(count_classes_samples_list) > 0:
                    zip_count_classes_samples_list = zip(list(pos_dom_num), count_classes_samples_list)
                    zip_count_classes_samples_list = list(reversed(sorted(zip_count_classes_samples_list, key=lambda t: t[1])))
                    print(zip_count_classes_samples_list)
                    dominant_class_inx = zip_count_classes_samples_list[dom_order - 1][0]
                    if zip_count_classes_samples_list[dom_order - 1][1] != 0:
                        new_records[i][dominant_field_inx] = self.keys_list[dominant_class_inx]
                    else:
                        new_records[i][dominant_field_inx] = 'no_dom_' + str(dom_order) + '_order'
                else:
                    new_records[i][dominant_field_inx] = ''
                if space_field_name is not None:
                    new_records[i][space_field_inx] = area_space
                if min_space > 0:
                    space = np.sum(np.array(count_classes_samples_list)) / len(y_pol_list) * 100
                    if space < min_space:
                        new_records[i][note_field_inx] = 'low space'
                # подсчет количества пикселей доминантных классов
                pix_count[np.where(pix_count == dominant_class_inx)] += len(X_pol_list)
        areas_with_dominant.records.extend(new_records)
        # сохранение shape-файла
        areas_with_dominant.save(dominant_shp_address)
        ## запись площадей доминантных классов
        #if space_table_address is not None:
        #    space_table = pd.DataFrame({'Colors': '' * len(pos_dom_keys),
        #                                'Pixels quantity': pix_count,
        #                                'Space (m^2)': pix_count * shot.pixel_space,
        #                                'Space (ha)': pix_count * shot.pixel_space / 10000,
        #                                }, index=pos_dom_keys,
        #                               columns=['Colors', 'Pixels quantity', 'Space (m^2)', 'Space (ha)'])
        #    # заполнение колонки colors цветом классов
        #    space_table = space_table.style.apply(colors_column, axis=None)
        #    # сохранение результата в виде таблицы excel по заданному адресу
        #    space_table.to_excel(space_table_address)

    def to_calc_dominant_2(self, areas_shp_address, dominant_shp_address, dominant_field_name,
                         possible_dom_classes=None, fields_to_copy=None,
                         dom_order=1, min_space=0, note_field=None, space_field_name=None):

        geo_trans = self.geo_trans
        projection_ref = self.projection_ref
        pos_dom_num = []
        if possible_dom_classes is None:
            pos_dom_keys = self.keys_list
            pos_dom_num = range(0, len(self.keys_list))
        else:
            pos_dom_keys = possible_dom_classes
            keys_np = np.array(self.keys_list)
            for key in possible_dom_classes:
                pos_dom_num.append(np.where(keys_np == key)[0][0])
        # загрузка shape-файла с диска для чтения и редактирования
        initial_areas = shapefile.Reader(areas_shp_address)
        # создание нового shape-файла, если его нет
        if os.path.exists(dominant_shp_address):
            areas_with_dominant = shapefile.Editor(dominant_shp_address)
        else:
            areas_with_dominant = shapefile.Writer(shapefile.POLYGON)
            # копирование содержимого в новый файл
        areas_with_dominant._shapes.extend(initial_areas.shapes())
        if fields_to_copy is None:
            new_records = initial_areas.records()
            areas_with_dominant.fields = list(initial_areas.fields)
        else:
            fields_to_copy_numbers = []
            fields_to_dominants_areas = [initial_areas.fields[0]]
            initial_areas_fields = np.array(initial_areas.fields)[:, 0][1:]
            for i, field in enumerate(initial_areas_fields):
                if field in fields_to_copy:
                    fields_to_copy_numbers.append(i)
                    fields_to_dominants_areas.append(initial_areas.fields[i + 1])
            new_records = []
            for i in fields_to_copy_numbers:
                new_records.append([row[i] for row in initial_areas.records()])
            new_records = list(map(list, zip(*new_records)))
            areas_with_dominant.fields = list(fields_to_dominants_areas)
        if space_field_name is not None:
            fields_names_list = np.array(areas_with_dominant.fields[1:])
            if space_field_name not in fields_names_list:
                areas_with_dominant.field(space_field_name)
                [pol.append('') for pol in new_records]
                space_field_inx = np.array(new_records).shape[1] - 1
            else:
                space_field_inx = np.where(fields_names_list == space_field_name)[0][0]
        # добавление поля для доминантного класса, если его еще нет
        fields_names_list = np.array(areas_with_dominant.fields[1:])
        if dominant_field_name not in fields_names_list:
            areas_with_dominant.field(dominant_field_name)
            [pol.append('') for pol in new_records]
            dominant_field_inx = np.array(new_records).shape[1] - 1
        else:
            dominant_field_inx = np.where(fields_names_list == dominant_field_name)[0][0]
        # добавление поля для примечания, если необходим
        note_field_inx = None
        if note_field is not None:
            fields_names_list = np.array(areas_with_dominant.fields[1:])
            if note_field not in fields_names_list:
                areas_with_dominant.field(note_field)
                [pol.append('') for pol in new_records]
                note_field_inx = np.array(new_records).shape[1] - 1
            else:
                note_field_inx = np.where(fields_names_list == note_field)[0][0]
        # вычисление доминанты для каждого полигона
        class_map = np.zeros((self.map.shape[0], self.map.shape[1]))
        class_map = class_map * np.nan
        for i in range(len(self.color_map)):
            class_map[np.where((self.map[:, :, 0] == self.color_map[i][0]) &
                               (self.map[:, :, 1] == self.color_map[i][1]) &
                               (self.map[:, :, 2] == self.color_map[i][2]) &
                               (self.map[:, :, 3] == self.color_map[i][3]))] = i
        pix_count = np.zeros(len(pos_dom_num))
        for i, shp in enumerate(initial_areas.shapes()):
            class_pol_list = []
            area_space = 0
            points_pix = np.array([list(i) for i in coord2pix(shp.points, geo_trans, projection_ref)])
            polygon_pix = geometry.Polygon(points_pix)
            x_pix_pol = np.int32(points_pix[:, 0])
            y_pix_pol = np.int32(points_pix[:, 1])
            up_pix = min(y_pix_pol)
            down_pix = max(y_pix_pol)
            left_pix = min(x_pix_pol)
            right_pix = max(x_pix_pol)
            pol_coord = []
            # отбор образцов из снимка, входящих в заданный полигон
            for y in range(up_pix, down_pix):
                for x in range(left_pix, right_pix):
                    if polygon_pix.contains(geometry.Point(x, y)):
                        area_space += 1
                        if not np.isnan(class_map[y, x]):
                            class_pol_list.append(class_map[y, x])
                            pol_coord.append([x, y])
                dominant_class_inx = None
                max_count_classes_samples = 0
                count_classes_samples_list = []
                for j in pos_dom_num:
                    keys_samples_count = len(np.where(np.array(class_pol_list) == j)[0])
                    count_classes_samples_list.append(keys_samples_count)
                    # if keys_samples_count > max_count_classes_samples:
                    #     max_count_classes_samples = keys_samples_count
                    #     dominant_class_inx = j
                if len(count_classes_samples_list) > 0:
                    zip_count_classes_samples_list = zip(list(pos_dom_num), count_classes_samples_list)
                    zip_count_classes_samples_list = list(reversed(sorted(zip_count_classes_samples_list, key=lambda t: t[1])))
                    print(zip_count_classes_samples_list)
                    dominant_class_inx = zip_count_classes_samples_list[dom_order - 1][0]
                    if zip_count_classes_samples_list[dom_order - 1][1] != 0:
                        new_records[i][dominant_field_inx] = self.keys_list[dominant_class_inx]
                    else:
                        new_records[i][dominant_field_inx] = 'no_dom_' + str(dom_order) + '_order'
                else:
                    new_records[i][dominant_field_inx] = ''
                if space_field_name is not None:
                    new_records[i][space_field_inx] = area_space
                if min_space > 0:
                    space = np.array(count_classes_samples_list)[dominant_class_inx] / len(class_pol_list) * 100
                    if space < min_space:
                        new_records[i][dominant_field_inx] = 'no_dom_' + str(dom_order) + '_order'
                # подсчет количества пикселей доминантных классов
                pix_count[np.where(pix_count == dominant_class_inx)] += len(class_pol_list)
        areas_with_dominant.records.extend(new_records)
        # сохранение shape-файла
        areas_with_dominant.save(dominant_shp_address)
        ## запись площадей доминантных классов
        #if space_table_address is not None:
        #    space_table = pd.DataFrame({'Colors': '' * len(pos_dom_keys),
        #                                'Pixels quantity': pix_count,
        #                                'Space (m^2)': pix_count * shot.pixel_space,
        #                                'Space (ha)': pix_count * shot.pixel_space / 10000,
        #                                }, index=pos_dom_keys,
        #                               columns=['Colors', 'Pixels quantity', 'Space (m^2)', 'Space (ha)'])
        #    # заполнение колонки colors цветом классов
        #    space_table = space_table.style.apply(colors_column, axis=None)
        #    # сохранение результата в виде таблицы excel по заданному адресу
        #    space_table.to_excel(space_table_address)

    def to_calc_right_percent(self, areas_shp_address, new_shp_address, right_percent_field_name, right_class_name,
                              fields_to_copy=None):
        geo_trans = self.geo_trans
        projection_ref = self.projection_ref
        pos_num = range(0, len(self.keys_list))
        # загрузка shape-файла с диска для чтения и редактирования
        initial_areas = shapefile.Reader(areas_shp_address)
        # создание нового shape-файла, если его нет
        if os.path.exists(new_shp_address):
            new_areas = shapefile.Editor(new_shp_address)
        else:
            new_areas = shapefile.Writer(shapefile.POLYGON)
            # копирование содержимого в новый файл
        new_areas._shapes.extend(initial_areas.shapes())
        if fields_to_copy is None:
            new_records = initial_areas.records()
            new_areas.fields = list(initial_areas.fields)
        else:
            fields_to_copy_numbers = []
            fields_to_new_areas = [initial_areas.fields[0]]
            initial_areas_fields = np.array(initial_areas.fields)[:, 0][1:]
            for i, field in enumerate(initial_areas_fields):
                if field in fields_to_copy:
                    fields_to_copy_numbers.append(i)
                    fields_to_new_areas.append(initial_areas.fields[i + 1])
            new_records = []
            for i in fields_to_copy_numbers:
                new_records.append([row[i] for row in initial_areas.records()])
            new_records = list(map(list, zip(*new_records)))
            new_areas.fields = list(fields_to_new_areas)
        # добавление поля для доминантного класса, если его еще нет
        fields_names_list = np.array(new_areas.fields[1:])
        if right_percent_field_name not in fields_names_list:
            new_areas.field(right_percent_field_name, fieldType="F")
            [pol.append(0) for pol in new_records]
            right_class_percent_field_inx = np.array(new_records).shape[1] - 1
        else:
            right_class_percent_field_inx = np.where(fields_names_list == right_percent_field_name)[0][0]
        right_class_name_field_inx = np.where(fields_names_list == right_class_name)[0][0]
        # вычисление доминанты для каждого полигона
        class_map = np.zeros((self.map.shape[0], self.map.shape[1]))
        class_map = class_map * np.nan
        for i in range(len(self.color_map)):
            class_map[np.where((self.map[:, :, 0] == self.color_map[i][0]) &
                               (self.map[:, :, 1] == self.color_map[i][1]) &
                               (self.map[:, :, 2] == self.color_map[i][2]) &
                               (self.map[:, :, 3] == self.color_map[i][3]))] = i
        pix_count = np.zeros(len(pos_num))
        for i, shp in enumerate(initial_areas.shapes()):
            class_pol_list = []
            area_space = 0
            points_pix = np.array([list(i) for i in coord2pix(shp.points, geo_trans, projection_ref)])
            polygon_pix = geometry.Polygon(points_pix)
            x_pix_pol = np.int32(points_pix[:, 0])
            y_pix_pol = np.int32(points_pix[:, 1])
            up_pix = min(y_pix_pol)
            down_pix = max(y_pix_pol)
            left_pix = min(x_pix_pol)
            right_pix = max(x_pix_pol)
            pol_coord = []
            # отбор образцов из снимка, входящих в заданный полигон
            for y in range(up_pix, down_pix):
                for x in range(left_pix, right_pix):
                    if polygon_pix.contains(geometry.Point(x, y)):
                        if not np.isnan(class_map[y, x]):
                            area_space += 1
                            class_pol_list.append(class_map[y, x])
                            pol_coord.append([x, y])
            right_class_inx = None

            count_classes_samples_list = []
            for j in pos_num:
                keys_samples_count = len(np.where(np.array(class_pol_list) == j)[0])
                count_classes_samples_list.append(keys_samples_count)
            if area_space > 0:
                right_class_key = initial_areas.records()[i][right_class_name_field_inx]
                right_class_num = list(np.where(np.array(self.keys_list) == right_class_key))[0][0]
                right_class_percent = count_classes_samples_list[right_class_num] / area_space * 100
                new_records[i][right_class_percent_field_inx] = right_class_percent
            else:
                new_records[i][right_class_percent_field_inx] = 0
            # подсчет количества пикселей доминантных классов
            pix_count[np.where(pix_count == right_class_inx)] += len(class_pol_list)
        new_areas.records.extend(new_records)
        # сохранение shape-файла
        new_areas.save(new_shp_address)

    def to_change_color_map(self, classes=None, color_map=None):
        if classes is not None:
            self.keys_list = classes
        if color_map is not None:
            color_map_rgba = []
            for color in color_map:
                if isinstance(color, str):
                    color_map_rgba.append(colors.to_rgba(color))
                else:
                    color_map_rgba.append(color)
            self.color_map = color_map_rgba

    def to_test_estimator_by_polygons(self, shp_address, real_attribute, file_name, directory_address):
        geo_trans = self.geo_trans
        projection_ref = self.projection_ref
        polygons = shapefile.Reader(shp_address)
        # вычисление доминанты для каждого полигона
        class_map = np.zeros((self.map.shape[0], self.map.shape[1]))
        class_map = class_map * np.nan
        for i in range(len(self.color_map)):
            class_map[np.where((self.map[:, :, 0] == self.color_map[i][0]) &
                               (self.map[:, :, 1] == self.color_map[i][1]) &
                               (self.map[:, :, 2] == self.color_map[i][2]) &
                               (self.map[:, :, 3] == self.color_map[i][3]))] = i
        real_attributes_names = np.array(polygons.fields)[:, 0]
        real_field_number = np.where(real_attributes_names == real_attribute)[0][0] - 1
        y_pred = []
        y_real = []
        for i, shp in enumerate(polygons.shapes()):
            real_class = list(np.where(np.array(self.keys_list) == polygons.record(i)[real_field_number]))[0][0]
            area_space = 0
            points_pix = np.array([list(i) for i in coord2pix(shp.points, geo_trans, projection_ref)])
            polygon_pix = geometry.Polygon(points_pix)
            x_pix_pol = np.int32(points_pix[:, 0])
            y_pix_pol = np.int32(points_pix[:, 1])
            up_pix = min(y_pix_pol)
            down_pix = max(y_pix_pol)
            left_pix = min(x_pix_pol)
            right_pix = max(x_pix_pol)
            # отбор образцов из снимка, входящих в заданный полигон
            for y in range(up_pix, down_pix):
                for x in range(left_pix, right_pix):
                    if polygon_pix.contains(geometry.Point(x, y)):
                        area_space += 1
                        if not np.isnan(class_map[y, x]):
                            y_pred.append(class_map[y, x])
                            y_real.append(real_class)
        self.to_compare(np.int64(y_real), np.int64(y_pred), file_name, directory_address=directory_address)


    def to_calc_mean(self, areas_shp_address, mean_shp_address, mean_field_name, shot, fields_to_copy=None,
                     spec_features=None, texture_features=None, texture_adjacency_directions_dict=None,
                     average_before_pred=False, dom_order=1, min_space=0, note_field=None, borders=None,
                     space_field_name=None):
        if type(shot) == list:
            hypercube = []
            geo_trans = []
            projection_ref = []
            for one_shot in shot:
                hypercube.append(one_shot.to_combine_data_in_hypercube(spec_features=spec_features,
                                                                       texture_features=texture_features,
                                                                       texture_adjacency_directions_dict=texture_adjacency_directions_dict))
                geo_trans.append(one_shot.spec_geo_trans)
                projection_ref.append(one_shot.spec_projection_ref)
        else:
            hypercube = shot.to_combine_data_in_hypercube(spec_features=spec_features,
                                                          texture_features=texture_features,
                                                          texture_dict=texture_adjacency_directions_dict)
            geo_trans = shot.spec_geo_trans
            projection_ref = shot.spec_projection_ref
        # загрузка shape-файла с диска для чтения и редактирования
        initial_areas = shapefile.Reader(areas_shp_address)
        # создание нового shape-файла, если его нет
        if os.path.exists(mean_shp_address):
            areas_to_mean = shapefile.Editor(mean_shp_address)
        else:
            areas_to_mean = shapefile.Writer(shapefile.POLYGON)
            # копирование содержимого в новый файл
            areas_to_mean._shapes.extend(initial_areas.shapes())
        if fields_to_copy is None:
            new_records = initial_areas.records()
            areas_to_mean.fields = list(initial_areas.fields)
        else:
            fields_to_copy_numbers = []
            fields_to_dominants_areas = [initial_areas.fields[0]]
            initial_areas_fields = np.array(initial_areas.fields)[:, 0][1:]
            for i, field in enumerate(initial_areas_fields):
                if field in fields_to_copy:
                    fields_to_copy_numbers.append(i)
                    fields_to_dominants_areas.append(initial_areas.fields[i + 1])
            new_records = []
            for i in fields_to_copy_numbers:
                new_records.append([row[i] for row in initial_areas.records()])
            new_records = list(map(list, zip(*new_records)))
            areas_to_mean.fields = list(fields_to_dominants_areas)
        # добавление поля для доминантного класса, если его еще нет
        fields_names_list = np.array(areas_to_mean.fields[1:])
        if mean_field_name not in fields_names_list:
            areas_to_mean.field(mean_field_name)
            [pol.append('') for pol in new_records]
            mean_field_inx = np.array(new_records).shape[1] - 1
        else:
            mean_field_inx = np.where(fields_names_list == mean_field_name)[0][0]
        if space_field_name is not None:
            fields_names_list = np.array(areas_to_mean.fields[1:])
            if space_field_name not in fields_names_list:
                areas_to_mean.field(space_field_name)
                [pol.append('') for pol in new_records]
                space_field_inx = np.array(new_records).shape[1] - 1
            else:
                space_field_inx = np.where(fields_names_list == space_field_name)[0][0]
        # добавление поля для примечания, если необходимо
        note_field_inx = None
        if note_field is not None:
            fields_names_list = np.array(areas_to_mean.fields[1:])
            if note_field not in fields_names_list:
                areas_to_mean.field(note_field)
                [pol.append('') for pol in new_records]
                note_field_inx = np.array(new_records).shape[1] - 1
            else:
                note_field_inx = np.where(fields_names_list == note_field)[0][0]
        # вычисление доминанты для каждого полигона
        current_hypercube = hypercube
        current_geo_trans = geo_trans
        current_projection_ref = projection_ref
        for i, shp in enumerate(initial_areas.shapes()):
            if type(hypercube) == list:
                current_hypercube = hypercube[i]
                current_geo_trans = geo_trans[i]
                current_projection_ref = projection_ref[i]
            X_pol_list = []
            points_pix = np.array([list(i) for i in coord2pix(shp.points, current_geo_trans, current_projection_ref)])

            polygon_pix = geometry.Polygon(points_pix)
            x_pix_pol = np.int32(points_pix[:, 0])
            y_pix_pol = np.int32(points_pix[:, 1])
            up_pix = min(y_pix_pol)
            down_pix = max(y_pix_pol)
            left_pix = min(x_pix_pol)
            right_pix = max(x_pix_pol)
            # отбор образцов из снимка, входящих в заданный полигон
            for y in range(up_pix, down_pix):
                for x in range(left_pix, right_pix):
                    if polygon_pix.contains(geometry.Point(x, y)):
                        X_pol_list.append(current_hypercube[:, y, x])
            if len(X_pol_list) > 0:
                X_pol_list = self.scaler.transform(np.array(X_pol_list))
            if len(X_pol_list) > 0:
                y_pol_list = self.mashine.predict(X_pol_list)
                if borders is not None:
                    y_pol_list = np.where(y_pol_list < borders[0], borders[0], y_pol_list)
                    y_pol_list = np.where(y_pol_list > borders[1], borders[1], y_pol_list)
                new_records[i][mean_field_inx] = np.mean(y_pol_list)
            else:
                new_records[i][mean_field_inx] = ''
            if space_field_name is not None:
                new_records[i][space_field_inx] = len(X_pol_list)
        areas_to_mean.records.extend(new_records)
        # сохранение shape-файла
        areas_to_mean.save(mean_shp_address)
        ## запись площадей доминантных классов
        #if space_table_address is not None:
        #    space_table = pd.DataFrame({'Colors': '' * len(pos_dom_keys),
        #                                'Pixels quantity': pix_count,
        #                                'Space (m^2)': pix_count * shot.pixel_space,
        #                                'Space (ha)': pix_count * shot.pixel_space / 10000,
        #                                }, index=pos_dom_keys,
        #                               columns=['Colors', 'Pixels quantity', 'Space (m^2)', 'Space (ha)'])
        #    # заполнение колонки colors цветом классов
        #    space_table = space_table.style.apply(colors_column, axis=None)
        #    # сохранение результата в виде таблицы excel по заданному адресу
        #    space_table.to_excel(space_table_address)

    def to_draw_classes(self, first_band_key, second_band_key, image_sample_set=None, uniform_samples=False,
                        hyperspaces_borders=(1, 0, 0, 1), plane_step=0.001):
        # выборка двумерных параметров и обучение с их помощью методом QDA
        duo_X_train, duo_y_train, duo_X_test, duo_y_test = to_prepare_selection(
            self.samples_set,
            features_keys_list=[first_band_key, second_band_key],
            non_classified=self.non_classified)
        self.test_mashine.fit(duo_X_train, duo_y_train)
        # отображение пространств класса
        # выборка, которая будет отображаться
        if image_sample_set is None:
            image_sample_set = self.samples_set
        X_image, y_image, X_test, y_test = to_prepare_selection(image_sample_set,
                                                                features_keys_list=[first_band_key, second_band_key])
        # двумерное пространство классов
        up_border = hyperspaces_borders[0]
        down_border = hyperspaces_borders[1]
        left_border = hyperspaces_borders[2]
        right_border = hyperspaces_borders[3]
        X_first, X_second = np.meshgrid(np.arange(down_border, up_border, plane_step),
                                        np.arange(left_border, right_border, plane_step))
        # вычисление пространств
        Z = self.test_mashine.predict(np.c_[X_first.ravel(), X_second.ravel()])
        Z = Z.reshape(X_first.shape)
        classes_colors = [colors.to_hex(self.color_map[i]) for i in range(0, Z.max() + 1)]
        lev = np.arange(Z.min() - 0.5, Z.max() + 1.5, 1)
        plt.contourf(X_first, X_second, Z, levels=lev, colors=classes_colors)
        # проставление точек
        for i in range(0, Z.max() + 1):
            class_indexes = np.where(y_image == i)
            plt.scatter(X_image[class_indexes, 0], X_image[class_indexes, 1], c=classes_colors[i], edgecolor='black',
                        label=self.keys_list[i])
        plt.legend()
        plt.xlabel(first_band_key)
        plt.ylabel(second_band_key)
        plt.show()

    def to_test_mashine(self, name=None, directory_address=None, test_sample_set=None, features=None):
        # выборка для тестирования
        if test_sample_set is None:
            X_test = self.X_test
            y_test = self.y_test
        else:
            X_train, y_train, X_test, y_test = to_prepare_selection(test_sample_set, features_keys_list=features,
                                                                    samples_keys=list(self.samples_set.sample_dict.keys()))
            X_test = self.scaler.transform(X_test)
        y_pred = self.mashine.predict(X_test)
        if self.samples_set.samples_type == 'classifier':
            return self.to_compare(y_test, y_pred, name, directory_address=directory_address)
        else:
            return metrics.mean_squared_error(y_test, y_pred)

    def self_test(self, name=None, directory_path=None, cross_validation=True, reclassification=True, sample_remainder=False,
                  cv=10, stratified=False):
        # функция, выделяющая дагональ таблицы (правильные результаты) и общих сумм (Total)
        def diagonal_excretion_and_total(table):
            style_table = table.copy()
            style_table.loc[:, :] = ''
            table_type = list(style_table.columns)[0][0]
            keys = list(style_table.loc['Predicted', (table_type, 'Real')])
            # выделение диагонали
            for key in keys:
                style_table.loc[('Predicted', key), (table_type, 'Real', key)] = 'color: green'
            # выделение Total
            style_table.loc[('Predicted', 'Total')] = 'background-color : yellow; color : black'
            style_table.loc[:, (table_type, 'Real', 'Total')] = 'background-color : yellow; color : black'
            style_table.loc[
                ('Predicted', 'Total'), (table_type, 'Real', 'Total')] = 'background-color : orange; color : black;'
            return style_table

        # функция, выделяющая значения больше нуля
        def up_zero_excretion(val):
            color = 'red' if val > 0 else 'black'
            return 'color: %s' % color

        conf_mat_data_frame_list = []
        acc_err_data_frame_list = []
        total_coef_data_frame_list = []

        if cross_validation:
            X_train, y_train, X_test, y_test = to_prepare_selection(self.samples_set,
                                                                    features_keys_list=self.features_keys,
                                                                    samples_keys=list(self.samples_set.sample_dict.keys()))
            X_train = self.scaler.transform(X_test)
            #y_train = self.mashine.predict(X_test)

            if stratified:
                kf = model_selection.KFold(cv, shuffle=True)
            else:
                kf = model_selection.StratifiedKFold(cv, shuffle=True)
            y_real = []
            y_pred = []
            for train_index, test_index in kf.split(X_train, y_train):
                X_cv_train = X_train[train_index]
                y_cv_train = y_train[train_index]
                X_cv_test = X_train[test_index]
                y_cv_test = y_train[test_index]
                self.test_mashine.fit(X_cv_train, y_cv_train)
                y_real += list(y_cv_test)
                y_pred += list(self.test_mashine.predict(X_cv_test))

            # формирование матрицы
            conf_mat_with_total = confusion_matrix(y_real, y_pred, self.keys_list)

            # вычисление общей точности с учетом возможности отказа от классификации
            overall_accuracy = metrics.accuracy_score(y_real, y_pred)
            # вычисление статистики Каппа
            cohen_kappa = metrics.cohen_kappa_score(y_real, y_pred)

            # определение ошибкок пропуска цели и ошибкок ложной тревоги, producer’s accuracy и user’s accuracy
            omission_err_list, commission_err_list, prod_accuracy_list, user_accuracy_list = errors_pack(
                conf_mat_with_total)

            # запись данных в таблицы
            # таблица для confusion matrix
            # заголовки
            type_name = str(cv) + '-fold cross-validation'
            labels = np.concatenate((self.keys_list, ['Total']))
            indexes = pd.MultiIndex.from_product([['Predicted'], labels])
            header = pd.MultiIndex.from_product([[type_name], ['Real'], labels])

            # запись матрицы в DataFrame
            conf_mat_data_frame = pd.DataFrame(np.array(conf_mat_with_total), index=indexes, columns=header)
            # форматирование: выделение диагонали таблицы (правильной классификации) зеленым и ошибок больше нуля
            #   (неправильной классификации) красным
            conf_mat_data_frame = conf_mat_data_frame.style.applymap(up_zero_excretion).apply(
                diagonal_excretion_and_total,
                axis=None)
            conf_mat_data_frame_list.append(conf_mat_data_frame)

            # таблица с точностью и ошибками для каждого класса
            header = pd.MultiIndex.from_product([[type_name],
                                                 ['Omission error', 'Commission error', 'Producer’s accuracy',
                                                  'User’s accuracy']])
            acc_err_data_frame = pd.DataFrame(
                np.array([omission_err_list, commission_err_list, prod_accuracy_list, user_accuracy_list]).T,
                columns=header,
                index=self.keys_list)
            acc_err_data_frame_list.append(acc_err_data_frame)

            # таблица с общими коэффициентами (также записывается ошибка кросс-валидации)
            header = pd.MultiIndex.from_product([[type_name], ['Value']])
            total_coef_data_frame = pd.DataFrame(np.array([overall_accuracy, cohen_kappa]).T,
                                                 columns=header,
                                                 index=['Accuracy', 'Kappa Coefficient'])
            total_coef_data_frame_list.append(total_coef_data_frame)

        if reclassification:
            X_train, y_train, X_test, y_test = to_prepare_selection(self.samples_set, features_keys_list=self.features_keys,
                                                                    samples_keys=list(
                                                                        self.samples_set.sample_dict.keys()))
            X_test = self.scaler.transform(X_test)
            y_pred = self.mashine.predict(X_test)

            # формирование матрицы
            conf_mat_with_total = confusion_matrix(y_test, y_pred, self.keys_list)

            # вычисление общей точности с учетом возможности отказа от классификации
            overall_accuracy = metrics.accuracy_score(y_test, y_pred)
            # вычисление статистики Каппа
            cohen_kappa = metrics.cohen_kappa_score(y_test, y_pred)

            # определение ошибкок пропуска цели и ошибкок ложной тревоги, producer’s accuracy и user’s accuracy
            omission_err_list, commission_err_list, prod_accuracy_list, user_accuracy_list = errors_pack(conf_mat_with_total)

            # запись данных в таблицы
            # таблица для confusion matrix
            # заголовки
            labels = np.concatenate((self.keys_list, ['Total']))
            indexes = pd.MultiIndex.from_product([['Predicted'], labels])
            header = pd.MultiIndex.from_product([['Reclassification'], ['Real'], labels])

            # запись матрицы в DataFrame
            conf_mat_data_frame = pd.DataFrame(np.array(conf_mat_with_total), index=indexes, columns=header)
            # форматирование: выделение диагонали таблицы (правильной классификации) зеленым и ошибок больше нуля
            #   (неправильной классификации) красным
            conf_mat_data_frame = conf_mat_data_frame.style.applymap(up_zero_excretion).apply(
                diagonal_excretion_and_total,
                axis=None)
            conf_mat_data_frame_list.append(conf_mat_data_frame)

            # таблица с точностью и ошибками для каждого класса
            header = pd.MultiIndex.from_product([['Reclassification'], ['Omission error', 'Commission error', 'Producer’s accuracy', 'User’s accuracy']])
            acc_err_data_frame = pd.DataFrame(np.array([omission_err_list, commission_err_list, prod_accuracy_list, user_accuracy_list]).T,
                                              columns=header,
                                              index=self.keys_list)
            acc_err_data_frame_list.append(acc_err_data_frame)

            # таблица с общими коэффициентами (также записывается ошибка кросс-валидации)
            header = pd.MultiIndex.from_product([['Reclassification'], ['Value']])
            total_coef_data_frame = pd.DataFrame(np.array([overall_accuracy, cohen_kappa]).T,
                                                 columns=header,
                                                 index= ['Accuracy', 'Kappa Coefficient'])
            total_coef_data_frame_list.append(total_coef_data_frame)

        if sample_remainder:
            X_train, y_train, X_test, y_test = to_prepare_selection(self.sample_remainder, features_keys_list=self.features_keys,
                                                                    samples_keys=list(
                                                                        self.sample_remainder.sample_dict.keys()))
            X_test = self.scaler.transform(X_test)
            y_pred = self.mashine.predict(X_test)

            # формирование матрицы
            conf_mat_with_total = confusion_matrix(y_test, y_pred, self.keys_list)

            # вычисление общей точности с учетом возможности отказа от классификации
            overall_accuracy = metrics.accuracy_score(y_test, y_pred)
            # вычисление статистики Каппа
            cohen_kappa = metrics.cohen_kappa_score(y_test, y_pred)

            # определение ошибкок пропуска цели и ошибкок ложной тревоги, producer’s accuracy и user’s accuracy
            omission_err_list, commission_err_list, prod_accuracy_list, user_accuracy_list = errors_pack(conf_mat_with_total)

            # запись данных в таблицы
            # таблица для confusion matrix
            # заголовки
            labels = np.concatenate((self.keys_list, ['Total']))
            indexes = pd.MultiIndex.from_product([['Predicted'], labels])
            header = pd.MultiIndex.from_product([['Remainder'], ['Real'], labels])

            # запись матрицы в DataFrame
            conf_mat_data_frame = pd.DataFrame(np.array(conf_mat_with_total), index=indexes, columns=header)
            # форматирование: выделение диагонали таблицы (правильной классификации) зеленым и ошибок больше нуля
            #   (неправильной классификации) красным
            conf_mat_data_frame = conf_mat_data_frame.style.applymap(up_zero_excretion).apply(
                diagonal_excretion_and_total,
                axis=None)
            conf_mat_data_frame_list.append(conf_mat_data_frame)

            # таблица с точностью и ошибками для каждого класса
            header = pd.MultiIndex.from_product([['Remainder'],
                                                 ['Omission error', 'Commission error', 'Producer’s accuracy',
                                                  'User’s accuracy']])
            acc_err_data_frame = pd.DataFrame(
                np.array([omission_err_list, commission_err_list, prod_accuracy_list, user_accuracy_list]).T,
                columns=header,
                index=self.keys_list)
            acc_err_data_frame_list.append(acc_err_data_frame)

            # таблица с общими коэффициентами (также записывается ошибка кросс-валидации)
            header = pd.MultiIndex.from_product([['Remainder'], ['Value']])
            total_coef_data_frame = pd.DataFrame(np.array([overall_accuracy, cohen_kappa]).T,
                                                 columns=header,
                                                 index=['Accuracy', 'Kappa Coefficient'])
            total_coef_data_frame_list.append(total_coef_data_frame)

        # сохранение результата в виде таблицы excel по заданному адресу
        if directory_path is not None:
            full_path = "".join([directory_path, '\\', name, '.xlsx'])
            table_counter = -1
            with pd.ExcelWriter(full_path) as writer:
                for i in range(len(total_coef_data_frame_list)):
                    table_counter += 1
                    total_coef_data_frame_list[i].to_excel(writer, sheet_name='Total coefficients', startrow=6 * table_counter)
                    acc_err_data_frame_list[i].to_excel(writer, sheet_name='Classes coefficients', startrow=(4 + len(self.keys_list)) * table_counter)
                    conf_mat_data_frame_list[i].to_excel(writer, sheet_name='Confusion matrix', startrow=(6 + len(self.keys_list)) * table_counter)


    def to_compare_areas(self, real_polygons_path, real_compared_attribute,
                         pred_polygons_path, pred_compared_attribute, id_attribute, name, directory_path, weights_col_name=None,
                         rename_lib=None):
        real_areas = shapefile.Reader(real_polygons_path)
        pred_areas = shapefile.Reader(pred_polygons_path)
        # Списки названий аттрибутов
        real_attributes_names = np.array(real_areas.fields)[:, 0]
        pred_attributes_names = np.array(pred_areas.fields)[:, 0]
        # Номер аттрибута id для обоих файлов
        real_id_number = np.where(real_attributes_names == id_attribute)[0][0] - 1
        pred_id_number = np.where(pred_attributes_names == id_attribute)[0][0] - 1
        # Номера сравниваемых аттрибутов
        real_compared_number = np.where(real_attributes_names == real_compared_attribute)[0][0] - 1
        pred_compared_number = np.where(pred_attributes_names == pred_compared_attribute)[0][0] - 1
        y_real = list(np.array(list(sorted(zip(np.array(real_areas.records())[:, real_id_number],
                                               np.array(real_areas.records())[:, real_compared_number]))))[:, 1])
        y_pred = list(np.array(list(sorted(zip(np.array(pred_areas.records())[:, pred_id_number],
                                               np.array(pred_areas.records())[:, pred_compared_number]))))[:, 1])

        if rename_lib is not None:
            for key in rename_lib.keys():
                y_pred[np.where(y_pred == key)] = rename_lib[key]

        # Использование весов, при необходимости
        if weights_col_name is None:
            weights = len(y_real) * [1]
        else:
            weights_number = np.where(real_attributes_names == weights_col_name)[0][0] - 1
            weights = list(np.array(list(sorted(zip(np.array(real_areas.records())[:, real_id_number],
                                                    np.array(real_areas.records())[:, weights_number]))))[:, 1])
        # Удаление лишних значений
        i = 0
        while i < len(y_real):
            if (y_real[i] not in self.keys_list) or (y_pred[i] not in self.keys_list):
                del y_real[i]
                del y_pred[i]
                del weights[i]
            else:
                i += 1
        # Замена названий на номера
        y_real = np.array(y_real)
        y_pred = np.array(y_pred)
        weights = np.float64(np.array(weights))
        for i, key in enumerate(self.keys_list):
            y_real = np.where(y_real == key, i, y_real)
            y_pred = np.where(y_pred == key, i, y_pred)

        self.to_compare(np.int64(y_real), np.int64(y_pred), name, np.int64(weights), directory_path)

    def to_compare_areas_err(self, real_polygons_address, real_compared_attribute,
                             pred_polygons_address, pred_compared_attribute, id_attribute, weights_col_name=None):
        real_areas = shapefile.Reader(real_polygons_address)
        pred_areas = shapefile.Reader(pred_polygons_address)
        # Списки названий аттрибутов
        real_attributes_names = np.array(real_areas.fields)[:, 0]
        pred_attributes_names = np.array(pred_areas.fields)[:, 0]
        # Номер аттрибута id для обоих файлов
        real_id_number = np.where(real_attributes_names == id_attribute)[0][0] - 1
        pred_id_number = np.where(pred_attributes_names == id_attribute)[0][0] - 1
        # Номера сравниваемых аттрибутов
        real_compared_number = np.where(real_attributes_names == real_compared_attribute)[0][0] - 1
        pred_compared_number = np.where(pred_attributes_names == pred_compared_attribute)[0][0] - 1
        y_real = list(np.array(list(zip(np.array(real_areas.records())[:, real_id_number],
                                               np.array(real_areas.records())[:, real_compared_number])))[:, 1])
        y_pred = list(np.array(list(zip(np.array(pred_areas.records())[:, pred_id_number],
                                               np.array(pred_areas.records())[:, pred_compared_number])))[:, 1])
        # Использование весов, при необходимости
        if weights_col_name is None:
            weights = len(y_real) * [1]
        else:
            weights_number = np.where(real_attributes_names == weights_col_name)[0][0] - 1
            weights = list(np.array(list(zip(np.array(real_areas.records())[:, real_id_number],
                                            np.array(real_areas.records())[:, weights_number])))[:, 1])
        # Удаление лишних значений
        i = 0
        while i < len(y_real):
            if (len(y_real[i]) == 0) or (len(y_pred[i]) == 0):
                del y_real[i]
                del y_pred[i]
                del weights[i]
            else:
                i += 1
        # Замена названий на номера
        y_real = np.float64(np.array(y_real))
        y_pred = np.float64(np.array(y_pred))
        weights = np.float64(np.array(weights))
        mean_abs_err = metrics.mean_absolute_error(y_real, y_pred, sample_weight=weights)
        mean_squar_err = metrics.mean_squared_error(y_real, y_pred, sample_weight=weights)
        return mean_abs_err, mean_squar_err

    def to_compare(self, y_real, y_pred, name, weights=None, directory_address=None):
        if weights is None:
            weights = len(y_real) * [1]
        # функция, выделяющая дагональ таблицы (правильные результаты) и общих сумм (Total)
        def diagonal_excretion_and_total(table):
            style_table = table.copy()
            style_table.loc[:, :] = ''
            keys = list(style_table.loc['Predicted', 'Real'])
            # выделение диагонали
            for key in keys:
                style_table.loc[('Predicted', key), ('Real', key)] = 'color: green'
            # выделение Total
            style_table.loc[('Predicted', 'Total')] = 'background-color : yellow; color : black'
            style_table.loc[:, ('Real', 'Total')] = 'background-color : yellow; color : black'
            style_table.loc[
                ('Predicted', 'Total'), ('Real', 'Total')] = 'background-color : orange; color : black;'
            return style_table

        # функция, выделяющая значения больше нуля
        def up_zero_excretion(val):
            color = 'red' if val > 0 else 'black'
            return 'color: %s' % color

        # формирование матрицы
        conf_mat_with_total = confusion_matrix(y_real, y_pred, self.keys_list)

         # вычисление общей точности с учетом возможности отказа от классификации
        overall_accuracy = metrics.accuracy_score(y_real, y_pred, sample_weight=weights)

        # вычисление статистики Каппа
        cohen_kappa = metrics.cohen_kappa_score(y_real, y_pred)

        # определение ошибкок пропуска цели и ошибкок ложной тревоги, producer’s accuracy и user’s accuracy
        omission_err_list, commission_err_list, prod_accuracy_list, user_accuracy_list = errors_pack(conf_mat_with_total)

        # запись данных в таблицы
        # таблица для confusion matrix
        # заголовки
        labels = np.concatenate((self.keys_list, ['Total']))
        indexes = pd.MultiIndex.from_product([['Predicted'], labels])
        header = pd.MultiIndex.from_product([['Real'], labels])
        # запись матрицы в DataFrame
        conf_mat_data_frame = pd.DataFrame(np.array(conf_mat_with_total), index=indexes, columns=header)
        # форматирование: выделение диагонали таблицы (правильной классификации) зеленым и ошибок больше нуля
        #   (неправильной классификации) красным
        conf_mat_data_frame = conf_mat_data_frame.style.applymap(up_zero_excretion).apply(diagonal_excretion_and_total,
                                                                                          axis=None)
        # таблица с точностью и ошибками для каждого класса
        acc_err_data_frame = pd.DataFrame({'Omission error': omission_err_list,
                                           'Commission error': commission_err_list,
                                           'Producer’s accuracy': prod_accuracy_list,
                                           'User’s accuracy': user_accuracy_list},
                                            index=self.keys_list)
        # таблица с общими коэффициентами (также записывается ошибка кросс-валидации)
        total_coef_data_frame = pd.DataFrame({'Accuracy': overall_accuracy,
                                              'Kappa Coefficient': cohen_kappa},
                                              index=['Value']).T
        # площади классов
        # функция, заполняющая цветами колонку colors
        def colors_column(table):
            colored_table = table.copy()
            colored_table[:] = ''
            for i in range(0, len(self.keys_list)):
                hex_color = colors.to_hex(self.color_map[i])
                colored_table['Colors'][i] = 'background-color: %s' % hex_color
                colored_table['Colors'][i] = 'background-color: %s' % hex_color
            return colored_table
        # реальные объекты
        classes_samples = []
        for key in self.keys_list:
            classes_samples.append(len(self.samples_set.sample_dict[key].samples))
        classes_samples = np.array(classes_samples)
        samp_header = ['Colors', 'Samples quantity']
        samp_table = pd.DataFrame({'Colors': '' * len(self.keys_list),
                                    'Samples quantity': classes_samples
                                    }, index=self.keys_list, columns=samp_header)
        # заполнение колонки colors цветом классов
        space_table = samp_table.style.apply(colors_column, axis=None)
        # сохранение результата в виде таблицы excel по заданному адресу
        if directory_address is not None:
            full_address = "".join([directory_address, '\\', name, '.xlsx'])
            with pd.ExcelWriter(full_address) as writer:
                total_coef_data_frame.to_excel(writer, sheet_name='Total coefficients')
                acc_err_data_frame.to_excel(writer, sheet_name='Classes coefficients')
                conf_mat_data_frame.to_excel(writer, sheet_name='Confusion matrix')
                space_table.to_excel(writer, sheet_name='Samples')
        return overall_accuracy

    def to_save_map(self, file_name, file_directory):
        # адрес сохраняемой карты
        if not os.path.exists(file_directory):
            os.makedirs(file_directory)
        file_address = "".join([file_directory, '\\',  file_name, '.tif'])
        # переформатирование карты
        reshaped_map = 255 * np.moveaxis(self.map, -1, 0)
        # создание файла
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(file_address, len(self.map[0]), len(self.map[:, 0]), 4, gdal.GDT_Byte)
        dataset.SetGeoTransform((self.geo_trans[0],
                                 self.geo_trans[1],
                                 self.geo_trans[2],
                                 self.geo_trans[3],
                                 self.geo_trans[4],
                                 self.geo_trans[5]))
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromWkt(self.projection_ref)
        dataset.SetProjection(spatial_reference.ExportToWkt())
        dataset.GetRasterBand(1).WriteArray(reshaped_map[0])
        dataset.GetRasterBand(2).WriteArray(reshaped_map[1])
        dataset.GetRasterBand(3).WriteArray(reshaped_map[2])
        dataset.GetRasterBand(4).WriteArray(reshaped_map[3])
        dataset.FlushCache()
        dataset = None

    def to_save_reg_map(self, file_name, file_directory, gradation=100):
        # адрес сохраняемой карты
        file_address = "".join([file_directory, '\\',  file_name, '.tif'])
        # переформатирование карты
        #reshaped_map = np.moveaxis(self.map, -1, 0)
        # создание файла
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(file_address, len(self.map[0]), len(self.map[:, 0]), 1, gdal.GDT_Byte)
        dataset.SetGeoTransform((self.geo_trans[0],
                                 self.geo_trans[1],
                                 self.geo_trans[2],
                                 self.geo_trans[3],
                                 self.geo_trans[4],
                                 self.geo_trans[5]))
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromWkt(self.projection_ref)
        dataset.SetProjection(spatial_reference.ExportToWkt())
        dataset.GetRasterBand(1).WriteArray(gradation * self.map)
        dataset.FlushCache()
        dataset = None

    def to_save_class_spaces(self, file_name, file_directory):
        # функция, заполняющая цветами колонку colors
        def colors_column(table):
            colored_table = table.copy()
            colored_table[:] = ''
            for i in range(0, len(self.keys_list)):
                hex_color = colors.to_hex(self.color_map[i])
                colored_table['Colors'][i] = 'background-color: %s' % hex_color
            return colored_table

        # адрес сохраняемой таблицы площадей
        file_address = "".join([file_directory, '\\', file_name, '.xlsx'])
        classes_pixels = []
        classes_spaces = []
        for i in range(0, len(self.keys_list)):
            classes_pixels.append(self.class_pixels_number[i])
            classes_spaces.append(self.class_pixels_number[i] * self.pixel_space)
        table = pd.DataFrame({'Colors': '' * len(self.keys_list),
                              'Pixels quantity': classes_pixels,
                              'Space (m^2)': classes_spaces}, index=self.keys_list)
        # заполнение колонки colors
        table = table.style.apply(colors_column, axis=None)
        # сохранение таблицы
        table.to_excel(file_address)

def to_prepare_selection(samples_set, features_keys_list=None, samples_keys=None, non_classified=False,
                         non_classified_random_points_number=1000, characteristics_borders_dict=None, classification=True):
    # список ключей, сортированных по алфавиту к образцам для выборки. Номер ключа в листе будет номером класса,
    #   используемым при обработке
    # while len(image_sample_set.classes_list) < len(self.samples_set.classes_list):
    #
    #     for class_sample in self.samples_set.classes_list:
    #         if class_sample not in list(image_sample_set.classes_list.keys()):
    #             image_sample_set.to_add_empty_class(class_sample, self.samples_set.classes_list[class_sample].color)
    samples_list = sorted(list(samples_set.sample_dict.keys()))
    if samples_keys is not None:
        keys_list = sorted(samples_keys)
    else:
        keys_list = samples_list

    # извлечение выборок из классов
    if samples_set.samples_type == 'classifier':
        classes_samples_list = []
        for samples_name in samples_list:
            samples = samples_set.sample_dict[samples_name].samples
            y_number = keys_list.index(samples_name)
            classes_samples_list.append([samples, y_number])
        # if uniform_samples is not None:
        #     if uniform_samples == 'Undersampling':
        #         # вычисление класса с минимальным количеством образцов
        #         samples_num = min([len(classes_samples_list[i][0]) for i in range(len(classes_samples_list))])
        #         # исключение случайных образцов для каждого класса до тех пор, пока количество образцов не будет одинаковым и
        #         #   равным min_class_num
        #     if uniform_samples == 'Oversampling':
        #         # вычисление класса с минимальным количеством образцов
        #         samples_num = max([len(classes_samples_list[i][0]) for i in range(len(classes_samples_list))])
        #         # исключение случайных образцов для каждого класса до тех пор, пока количество образцов не будет одинаковым и
        #         #   равным min_class_num
        #
        #     if type(uniform_samples) is int:
        #         samples_num = uniform_samples
        #     classes_samples_list, sample_remainder = sampling(classes_samples_list, samples_num)
        # else:
        #     sample_remainder = len(classes_samples_list) * [[[]]]

        # формирование обучающей выборки
        if features_keys_list is None:
            features_keys_list = list(samples_set.features_dict.keys())
        features_dict = samples_set.features_dict
        bands_numbers = []
        for feature_key in features_keys_list:
            bands_numbers.append(features_dict[feature_key])
        X_train = np.empty((0, len(bands_numbers)), int)
        #   значения классов для выборки
        y_train = []
        for y_number in range(0, len(classes_samples_list)):
            if classes_samples_list[y_number][0] != []:
                current_class_selection = np.array(classes_samples_list[y_number][0])[:, bands_numbers]
                X_train = np.append(X_train, current_class_selection, axis=0)
                y_train = np.append(y_train, np.array([classes_samples_list[y_number][1]] * len(current_class_selection)), axis=0)
        if y_train != []:
            y_train = y_train.astype(dtype='int32')
        # тестовая выборки для тестирования качества выборки (без non-classified)
        X_test = X_train
        y_test = y_train
    else:
        # формирование обучающей выборки
        if features_keys_list is None:
            features_keys_list = list(samples_set.features_dict.keys())
        features_dict = samples_set.features_dict
        bands_numbers = []
        for feature_key in features_keys_list:
            bands_numbers.append(features_dict[feature_key])
        X_train = []
        #   значения классов для выборки
        y_train = []
        for X_y in samples_set.sample_dict['regression'].samples:
            X_train.append(np.array(X_y[0])[bands_numbers])
            y_train.append(X_y[1])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        # if uniform_samples == 'Oversampling':
        #     classes_list = list(set(y_train))
        #     # вычисление класса с минимальным количеством образцов
        #     max_class_num = max([len(np.where(y_train == classes_list[i])[0]) for i in range(len(classes_list))])
        #     # исключение случайных образцов для каждого класса до тех пор, пока количество образцов не будет одинаковым и
        #     #   равным min_class_num
        #     for clas in classes_list:
        #         original_y = y_train[np.where(y_train == clas)]
        #         original_sample = X_train[np.where(y_train == clas)]
        #         while (len(np.where(y_train == clas)[0]) < max_class_num):
        #             if max_class_num - len(np.where(y_train == clas)[0] >= len(original_y)):
        #                 X_train = np.concatenate([X_train, original_sample])
        #                 y_train = np.concatenate([y_train, original_y])
                    # else:
                    #     classes_samples[0] += list(np.array(original_sample)[np.random.choice(len(original_sample),
                    #                                                                           max_class_num -
                    #                                                                           len(classes_samples[0]),
                    #                                                                           replace=False)])
        # тестовая выборки для тестирования качества выборки (без non-classified)
        X_test = X_train
        y_test = y_train
    #   добавление равномерно распределенных по пространству параметров неклассифицируемых объектов
    if non_classified:
        # расставление границ для характеристик. Если они не заданы для характеристики, то от 0 до 1
        if characteristics_borders_dict is None:
            characteristics_borders_dict = {}
        characteristics_borders_list = []
        for feature_key in list(features_dict.keys()):
            borders = characteristics_borders_dict.get(feature_key)
            if borders is not None:
                characteristics_borders_list.append(characteristics_borders_dict.get(feature_key))
            else:
                characteristics_borders_list.append((0, 1))
        # формирование выборки класса non-classification из случайных точек
        non_classified_samples = [np.random.uniform(border[0], border[1], non_classified_random_points_number)
                                  for border in characteristics_borders_list]
        non_classified_samples = np.swapaxes(non_classified_samples, 0, 1)
        non_classified_train_samples = np.array(non_classified_samples)[:, bands_numbers]
        X_train = np.append(X_train, non_classified_train_samples, axis=0)
        y_train = np.append(y_train, np.array([len(classes_samples_list)] * len(non_classified_train_samples)),
                            axis=0)
    return X_train, y_train, X_test, y_test

def sampling(samples_set, samples_num):
    if type(samples_num) != list:
        samples_num = [samples_num] * len(list(samples_set.sample_dict.keys()))
    sample_remainder = copy.deepcopy(samples_set)
    new_samples_set = copy.deepcopy(samples_set)
    keys_list = sorted(list(sample_remainder.sample_dict.keys()))
    for key in keys_list:
        sample_remainder.sample_dict[key].samples = []
    for i, key in enumerate(keys_list):
        if len(new_samples_set.sample_dict[key].samples) > samples_num[i]:
            while (len(new_samples_set.sample_dict[key].samples) > samples_num[i]):
                inx = random.randint(0, len(new_samples_set.sample_dict[key].samples) - 1)
                sample_remainder.sample_dict[key].samples.append(new_samples_set.sample_dict[key].samples[inx])
                del new_samples_set.sample_dict[key].samples[inx]

        if len(new_samples_set.sample_dict[key].samples) < samples_num[i]:
            original_sample = copy.deepcopy(new_samples_set.sample_dict[key].samples)
            while (len(new_samples_set.sample_dict[key].samples) < samples_num[i]):
                if samples_num[i] - len(new_samples_set.sample_dict[key].samples) >= len(original_sample):
                    new_samples_set.sample_dict[key].samples += original_sample
                else:
                    new_samples_set.sample_dict[key].samples += list(np.array(original_sample)[np.random.choice(len(original_sample),
                                                                                          samples_num[i] -
                                                                                          len(new_samples_set.sample_dict[key].samples),
                                                                                          replace=False)])
    return new_samples_set, sample_remainder

def confusion_matrix(y_real, y_pred, keys_list):
    # формирование матрицы
    conf_mat = metrics.confusion_matrix(y_real, y_pred).T
    unique_values = np.unique(np.concatenate((y_pred, y_real)))
    if len(unique_values) < len(keys_list):
        for i in range(0, len(keys_list)):
            if i not in unique_values:
                conf_mat = np.insert(np.insert(conf_mat, i, np.zeros((1, len(conf_mat))), axis=0), i,
                                     np.zeros((1, len(conf_mat) + 1)), axis=1)

    # добавление сумм
    conf_mat_with_total = np.concatenate((conf_mat, np.zeros((1, len(conf_mat)))))
    conf_mat_with_total = np.concatenate((conf_mat_with_total, np.zeros((len(conf_mat_with_total), 1))), axis=1)
    conf_mat_with_total[:-1, -1] = np.sum(conf_mat_with_total[:-1, :-1], axis=1)
    conf_mat_with_total[-1, :] = np.sum(conf_mat_with_total[:-1, :], axis=0)

    return conf_mat_with_total

def errors_pack(conf_mat_with_total):
    omission_err_list = []
    commission_err_list = []
    prod_accuracy_list = []
    user_accuracy_list = []

    conf_mat = conf_mat_with_total[:-1, :-1]
    for i in range(0, len(conf_mat)):
        err_ind = [j != i for j in range(0, len(conf_mat))]
        om_err = sum(conf_mat[err_ind, i]) / float(conf_mat_with_total[-1, i])
        omission_err_list.append(om_err)
        com_err = sum(conf_mat[i, err_ind]) / float(conf_mat_with_total[i, -1])
        commission_err_list.append(com_err)
        prod_acc = conf_mat[i, i] / float(conf_mat_with_total[-1, i])
        prod_accuracy_list.append(prod_acc)
        user_acc = conf_mat[i, i] / float(conf_mat_with_total[i, -1])
        user_accuracy_list.append(user_acc)
    return omission_err_list, commission_err_list, prod_accuracy_list, user_accuracy_list

def add_del_rf(samples_set, test_samples=None, uniform_samples=False, non_classified=False, non_classified_color='black',
               classes_borders=None, non_classified_random_points_number=1000,
               potential_features=None, first_features=None, score='accuracy', cv_=10,
               greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10,
               n_estimators=(10, 20), max_depth=(3, 5), class_weight=('balanced', 'balanced_subsample'),
               del_algorithm=True, luft=0, compare_with_first=False, to_plot_history=False):
    if potential_features is None:
        current_potential_features = list(samples_set.features_dict.keys())
    else:
        current_potential_features = copy.deepcopy(potential_features)
    if first_features is None:
        first_features = []
    add_features_keys = []
    if compare_with_first or len(first_features) == 0:
        best_score = 0.0
    else:
        map = Map(samples_set, first_features, uniform_samples=uniform_samples,
                  non_classified=non_classified, non_classified_color=non_classified_color,
                  classes_borders=classes_borders,
                  non_classified_random_points_number=non_classified_random_points_number)
        test_estimator = map.to_fit_by_RF(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                                          k_fold_order=k_fold_order, n_estimators=n_estimators,
                                          max_depth=max_depth, class_weight=class_weight)
        if test_samples is not None:
            best_score = map.to_test_mashine(test_sample_set=test_samples, features=first_features)
        else:
            best_score = model_selection.cross_validate(test_estimator, map.X_test, map.y_test,
                                                         scoring=score, cv=cv_)['test_score'].mean()
    history = []
    # add
    while True:
        feature_inx_max_score = (None, 0, 0.0)
        for i, pot_feature_key in enumerate(current_potential_features):
            current_features_keys = first_features + add_features_keys + [pot_feature_key]
            map = Map(samples_set, current_features_keys, uniform_samples=uniform_samples,
                      non_classified=non_classified, non_classified_color=non_classified_color,
                      classes_borders=classes_borders,
                      non_classified_random_points_number=non_classified_random_points_number)
            test_estimator = map.to_fit_by_RF(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                                              k_fold_order=k_fold_order, n_estimators=n_estimators,
                                               max_depth=max_depth, class_weight=class_weight)
            if test_samples is not None:
                score_value = map.to_test_mashine(test_sample_set=test_samples, features=current_features_keys,
                                                  uniform_samples=True)
            else:
                score_value = model_selection.cross_validate(test_estimator, map.X_test, map.y_test,
                                                             scoring=score, cv=cv_)['test_score'].mean()
            if score_value > feature_inx_max_score[2]:
                feature_inx_max_score = (pot_feature_key, i, score_value)
        if feature_inx_max_score[2] - best_score > luft:
            best_score = feature_inx_max_score[2]
            add_features_keys.append(feature_inx_max_score[0])
            del current_potential_features[feature_inx_max_score[1]]
            history_note = [copy.deepcopy(add_features_keys), best_score]
            history.append(history_note)
        else:
            break
    # del
    if del_algorithm:
        while True:
            del_feature_inx_max_score = (None, 0, 0.0)
            for i, pot_del_feature_key in enumerate(add_features_keys):
                current_features_keys = first_features + add_features_keys
                del current_features_keys[i]
                map = Map(samples_set, current_features_keys, uniform_samples=uniform_samples,
                          non_classified=non_classified, non_classified_color=non_classified_color,
                          classes_borders=classes_borders,
                          non_classified_random_points_number=non_classified_random_points_number)
                test_estimator = map.to_fit_by_RF(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                                                  k_fold_order=k_fold_order, n_estimators=n_estimators,
                                                  max_depth=max_depth, class_weight=class_weight)
                cv_score_value = model_selection.cross_validate(test_estimator, map.X_test, map.y_test,
                                                                scoring=score, cv=cv_)['test_score'].mean()
                if cv_score_value > del_feature_inx_max_score[2]:
                    del_feature_inx_max_score = (pot_del_feature_key, i, cv_score_value)
            if feature_inx_max_score[2] - best_score > luft and len(add_features_keys) > 1:
                best_score = del_feature_inx_max_score[2]
                del add_features_keys[del_feature_inx_max_score[1]]
                history_note = [copy.deepcopy(add_features_keys), best_score]
                history.append(history_note)
            else:
                break
    if to_plot_history:
        history_score = np.array(history)[:, 1]
        plt.plot(range(len(history_score)), history_score)
        plt.show()
    return add_features_keys, best_score, history

def add_del_knn(samples_set, test_samples=None, uniform_samples=False, non_classified=False, non_classified_color='black',
               classes_borders=None, non_classified_random_points_number=1000,
               potential_features=None, first_features=None, score='accuracy', cv_=10,
               greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10,
               n_neighbors=(5, 10), weights=('uniform', 'distance'),
               algorithm=('auto', 'ball_tree', 'kd_tree', 'brute'), p=(2, 1), del_algorithm=True, luft=0, to_plot_history=False):
    if potential_features is None:
        current_potential_features = list(samples_set.features_dict.keys())
    else:
        current_potential_features = copy.deepcopy(potential_features)
    if first_features is None:
        first_features = []
    add_features_keys = []
    best_score = 0.0
    history = []
    # add
    while True:
        feature_inx_max_score = (None, 0, 0.0)
        for i, pot_feature_key in enumerate(current_potential_features):
            current_features_keys = first_features + add_features_keys + [pot_feature_key]
            map = Map(samples_set, current_features_keys, uniform_samples=uniform_samples,
                      non_classified=non_classified, non_classified_color=non_classified_color,
                      classes_borders=classes_borders,
                      non_classified_random_points_number=non_classified_random_points_number)
            test_estimator = map.to_fit_by_KNN(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                                               k_fold_order=k_fold_order, n_neighbors=n_neighbors, weights=weights,
                                               algorithm=algorithm, p=p)
            if test_samples is not None:
                score_value = map.to_test_mashine(test_sample_set=test_samples, features=current_features_keys,
                                                  uniform_samples=True)
            else:
                score_value = model_selection.cross_validate(test_estimator, map.X_test, map.y_test,
                                                                scoring=score, cv=cv_)['test_score'].mean()
            if score_value > feature_inx_max_score[2]:
                feature_inx_max_score = (pot_feature_key, i, score_value)
            print(score_value, current_features_keys)
        if feature_inx_max_score[2] - best_score > luft:
            best_score = feature_inx_max_score[2]
            add_features_keys.append(feature_inx_max_score[0])
            del current_potential_features[feature_inx_max_score[1]]
            history_note = [copy.deepcopy(add_features_keys), best_score]
            history.append(history_note)
        else:
            break
    # del
    if del_algorithm:
        while True:
            del_feature_inx_max_score = (None, 0, 0.0)
            for i, pot_del_feature_key in enumerate(add_features_keys):
                current_features_keys = first_features + add_features_keys
                del current_features_keys[i]
                map = Map(samples_set, current_features_keys, uniform_samples=uniform_samples,
                          non_classified=non_classified, non_classified_color=non_classified_color,
                          classes_borders=classes_borders,
                          non_classified_random_points_number=non_classified_random_points_number)
                test_estimator = map.to_fit_by_KNN(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                                                   k_fold_order=k_fold_order, n_neighbors=n_neighbors, weights=weights,
                                                   algorithm=algorithm, p=p)
                if test_samples is not None:
                    score_value = map.to_test_mashine(test_sample_set=test_samples, features=current_features_keys,
                                                      uniform_samples=True)
                else:
                    score_value = model_selection.cross_validate(test_estimator, map.X_test, map.y_test,
                                                                scoring=score, cv=cv_)['test_score'].mean()
                if score_value > del_feature_inx_max_score[2]:
                    del_feature_inx_max_score = (pot_del_feature_key, i, score_value)
            if feature_inx_max_score[2] - best_score > luft and len(add_features_keys) > 1:
                best_score = del_feature_inx_max_score[2]
                del add_features_keys[del_feature_inx_max_score[1]]
                history_note = [copy.deepcopy(add_features_keys), best_score]
                history.append(history_note)
            else:
                break
    if to_plot_history:
        history_score = np.array(history)[:, 1]
        plt.plot(range(len(history_score)), history_score)
        plt.show()
    return add_features_keys, best_score, history

def add_del_svm(samples_set, uniform_samples=False, non_classified=False, non_classified_color='black',
               classes_borders=None, non_classified_random_points_number=1000,
               potential_features=None, first_features=None, score='accuracy', cv_=10,
               greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10,
               C=(1., 10.), degree=(3, 4), kernel=('rbf', 'linear', 'poly', 'sigmond', 'precomputed'),
               gamma=('auto', 'scale'), shrinking=(True, False), class_weight=(None, 'balanced'),
               decision_function_shape=('ovo', 'ovr'), del_algorithm=True, luft=0, to_plot_history=False):
    if potential_features is None:
        current_potential_features = list(samples_set.features_dict.keys())
    else:
        current_potential_features = copy.deepcopy(potential_features)
    if first_features is None:
        first_features = []
    add_features_keys = []
    best_score = 0.0
    history = []
    # add
    while True:
        feature_inx_max_score = (None, 0, 0.0)
        for i, pot_feature_key in enumerate(current_potential_features):
            current_features_keys = first_features + add_features_keys + [pot_feature_key]
            map = Map(samples_set, current_features_keys, uniform_samples=uniform_samples,
                      non_classified=non_classified, non_classified_color=non_classified_color,
                      classes_borders=classes_borders,
                      non_classified_random_points_number=non_classified_random_points_number)
            test_estimator = map.to_fit_by_SVM(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                                               k_fold_order=k_fold_order,
                                               C=C, degree=degree, kernel=kernel,
                                               gamma=gamma, shrinking=shrinking, class_weight=class_weight,
                                               decision_function_shape=decision_function_shape)
            cv_score_value = model_selection.cross_validate(test_estimator, map.X_test, map.y_test,
                                                            scoring=score, cv=cv_)['test_score'].mean()
            if cv_score_value > feature_inx_max_score[2]:
                feature_inx_max_score = (pot_feature_key, i, cv_score_value)
        if feature_inx_max_score[2] - best_score > luft:
            best_score = feature_inx_max_score[2]
            add_features_keys.append(feature_inx_max_score[0])
            del current_potential_features[feature_inx_max_score[1]]
            history_note = [copy.deepcopy(add_features_keys), best_score]
            history.append(history_note)
        else:
            break
    # del
    if del_algorithm:
        while True:
            del_feature_inx_max_score = (None, 0, 0.0)
            for i, pot_del_feature_key in enumerate(add_features_keys):
                current_features_keys = first_features + add_features_keys
                del current_features_keys[i]
                map = Map(samples_set, current_features_keys, uniform_samples=uniform_samples,
                          non_classified=non_classified, non_classified_color=non_classified_color,
                          classes_borders=classes_borders,
                          non_classified_random_points_number=non_classified_random_points_number)
                test_estimator = map.to_fit_by_SVM(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                                                   k_fold_order=k_fold_order,
                                                   C=C, degree=degree, kernel=kernel,
                                                   gamma=gamma, shrinking=shrinking, class_weight=class_weight,
                                                   decision_function_shape=decision_function_shape)
                cv_score_value = model_selection.cross_validate(test_estimator, map.X_test, map.y_test,
                                                                scoring=score, cv=cv_)['test_score'].mean()
                if cv_score_value > del_feature_inx_max_score[2]:
                    del_feature_inx_max_score = (pot_del_feature_key, i, cv_score_value)
            if feature_inx_max_score[2] - best_score > luft and len(add_features_keys) > 1:
                best_score = del_feature_inx_max_score[2]
                del add_features_keys[del_feature_inx_max_score[1]]
                history_note = [copy.deepcopy(add_features_keys), best_score]
                history.append(history_note)
            else:
                break
    if to_plot_history:
        history_score = np.array(history)[:, 1]
        plt.plot(range(len(history_score)), history_score)
        plt.show()
    return add_features_keys, best_score, history

def add_del_qda(samples_set, uniform_samples=False, non_classified=False, non_classified_color='black',
               classes_borders=None, non_classified_random_points_number=1000,
               potential_features=None, first_features=None, score='accuracy', cv_=10,
               greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10,
               reg_param=(0, -0.1, 0.1), del_algorithm=True, luft=0, to_plot_history=False):
    if potential_features is None:
        current_potential_features = list(samples_set.features_dict.keys())
    else:
        current_potential_features = copy.deepcopy(potential_features)
    if first_features is None:
        first_features = []
    add_features_keys = []
    best_score = 0.0
    history = []
    # add
    while True:
        feature_inx_max_score = (None, 0, 0.0)
        for i, pot_feature_key in enumerate(current_potential_features):
            current_features_keys = first_features + add_features_keys + [pot_feature_key]
            map = Map(samples_set, current_features_keys, uniform_samples=uniform_samples,
                      non_classified=non_classified, non_classified_color=non_classified_color,
                      classes_borders=classes_borders,
                      non_classified_random_points_number=non_classified_random_points_number)
            test_estimator = map.to_fit_by_QDA(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                                               k_fold_order=k_fold_order, reg_param=reg_param)
            cv_score_value = model_selection.cross_validate(test_estimator, map.X_test, map.y_test,
                                                            scoring=score, cv=cv_)['test_score'].mean()
            if cv_score_value > feature_inx_max_score[2]:
                feature_inx_max_score = (pot_feature_key, i, cv_score_value)
        if feature_inx_max_score[2] - best_score > luft:
            best_score = feature_inx_max_score[2]
            add_features_keys.append(feature_inx_max_score[0])
            del current_potential_features[feature_inx_max_score[1]]
            history_note = [copy.deepcopy(add_features_keys), best_score]
            history.append(history_note)
        else:
            break
    # del
    if del_algorithm:
        while True:
            del_feature_inx_max_score = (None, 0, 0.0)
            for i, pot_del_feature_key in enumerate(add_features_keys):
                current_features_keys = first_features + add_features_keys
                del current_features_keys[i]
                map = Map(samples_set, current_features_keys, uniform_samples=uniform_samples,
                          non_classified=non_classified, non_classified_color=non_classified_color,
                          classes_borders=classes_borders,
                          non_classified_random_points_number=non_classified_random_points_number)
                test_estimator = map.to_fit_by_QDA(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                                                   k_fold_order=k_fold_order, reg_param=reg_param)
                cv_score_value = model_selection.cross_validate(test_estimator, map.X_test, map.y_test,
                                                                scoring=score, cv=cv_)['test_score'].mean()
                if cv_score_value > del_feature_inx_max_score[2]:
                    del_feature_inx_max_score = (pot_del_feature_key, i, cv_score_value)
            if feature_inx_max_score[2] - best_score > luft and len(add_features_keys) > 1:
                best_score = del_feature_inx_max_score[2]
                del add_features_keys[del_feature_inx_max_score[1]]
                history_note = [copy.deepcopy(add_features_keys), best_score]
                history.append(history_note)
            else:
                break
    if to_plot_history:
        history_score = np.array(history)[:, 1]
        plt.plot(range(len(history_score)), history_score)
        plt.show()
    return add_features_keys, best_score, history

def rand_prif_rf(samples_set, iter_num, test_samples=None, uniform_samples=False, non_classified=False,
                 non_classified_color='black',
                 classes_borders=None, non_classified_random_points_number=1000,
                 potential_features=None, first_features=None, score='accuracy', cv_=10,
                 greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10,
                 n_estimators=(10, 20), max_depth=(3, 5), class_weight=('balanced', 'balanced_subsample'),
                 luft=0, to_plot_history=False):
    if potential_features is None:
        current_potential_features = list(samples_set.features_dict.keys())
    else:
        current_potential_features = potential_features
    if first_features is None:
        first_features = []
    best_comp = []
    best_score = 0.0
    feature_max_score = (None, 0.0)
    history = [[[], 0.0]]
    # вероятности для колличества признаков в комбинации
    comb_quan_list = []
    features_quan = len(current_potential_features)
    for i in range(0, features_quan):
        comb_quan = math.factorial(features_quan) / (math.factorial(features_quan - (i + 1)) * math.factorial(i + 1))
        comb_quan_list.append(comb_quan)
    prob_num_arr = np.array(comb_quan_list) / np.sum(np.array(comb_quan_list))
    for i in range(0, iter_num):
        # Отбор комбинации
        previous_comp = np.array(history)[:, 0]
        while True:
            rand = np.random.random()
            sum_rand = 0
            j = 0
            while True:
                sum_rand += prob_num_arr[j]
                if sum_rand >= rand:
                    break
                else:
                    j += 1
            add_features_keys = np.sort(np.random.choice(current_potential_features, j + 1, replace=False))
            if add_features_keys not in previous_comp:
                break
        current_features_keys = first_features + list(add_features_keys)
        map = Map(samples_set, current_features_keys, uniform_samples=uniform_samples,
                  non_classified=non_classified, non_classified_color=non_classified_color,
                  classes_borders=classes_borders,
                  non_classified_random_points_number=non_classified_random_points_number)
        test_estimator = map.to_fit_by_RF(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                                          k_fold_order=k_fold_order, n_estimators=n_estimators,
                                          max_depth=max_depth, class_weight=class_weight)
        if test_samples is not None:
            score_value = map.to_test_mashine(test_sample_set=test_samples, features=current_features_keys,
                                              uniform_samples=True)
        else:
            score_value = model_selection.cross_validate(test_estimator, map.X_test, map.y_test,
                                                         scoring=score, cv=cv_)['test_score'].mean()
        history_note = [list(copy.deepcopy(add_features_keys)), score_value]
        history.append(history_note)
        if score_value > feature_max_score[1]:
            feature_max_score = (current_features_keys, score_value)
        if feature_max_score[1] - best_score > luft:
            best_score = feature_max_score[1]
            best_comp = current_features_keys
    history = list(np.array(history)[1:])
    if to_plot_history:
        history_score = np.array(history)[:, 1]
        plt.plot(range(len(history_score)), history_score)
        plt.show()
    return best_comp, best_score#, history

def rand_prif_knn(samples_set, iter_num, test_samples=None, uniform_samples=False, non_classified=False,
                 non_classified_color='black',
                 classes_borders=None, non_classified_random_points_number=1000,
                 potential_features=None, first_features=None, score='accuracy', cv_=10,
                 greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10,
                  n_neighbors=(5, 10), weights=('uniform', 'distance'),
                  algorithm=('auto', 'ball_tree', 'kd_tree', 'brute'), p=(2, 1),
                 luft=0, to_plot_history=False):
    if potential_features is None:
        current_potential_features = list(samples_set.features_dict.keys())
    else:
        current_potential_features = potential_features
    if first_features is None:
        first_features = []
    best_comp = []
    best_score = 0.0
    feature_max_score = (None, 0.0)
    history = [[[], 0.0]]
    # вероятности для колличества признаков в комбинации
    comb_quan_list = []
    features_quan = len(current_potential_features)
    for i in range(0, features_quan):
        comb_quan = math.factorial(features_quan) / (math.factorial(features_quan - (i + 1)) * math.factorial(i + 1))
        comb_quan_list.append(comb_quan)
    prob_num_arr = np.array(comb_quan_list) / np.sum(np.array(comb_quan_list))
    for i in range(0, iter_num):
        # Отбор комбинации
        previous_comp = np.array(history)[:, 0]
        while True:
            rand = np.random.random()
            sum_rand = 0
            j = 0
            while True:
                sum_rand += prob_num_arr[j]
                if sum_rand >= rand:
                    break
                else:
                    j += 1
            add_features_keys = np.sort(np.random.choice(current_potential_features, j + 1, replace=False))
            if add_features_keys not in previous_comp:
                break
        current_features_keys = first_features + list(add_features_keys)
        map = Map(samples_set, current_features_keys, uniform_samples=uniform_samples,
                  non_classified=non_classified, non_classified_color=non_classified_color,
                  classes_borders=classes_borders,
                  non_classified_random_points_number=non_classified_random_points_number)
        test_estimator = map.to_fit_by_KNN(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                                          k_fold_order=k_fold_order, n_neighbors=n_neighbors, weights=weights,
                                          algorithm=algorithm, p=p)
        if test_samples is not None:
            score_value = map.to_test_mashine(test_sample_set=test_samples, features=current_features_keys,
                                              uniform_samples=False
                                              )
        else:
            score_value = model_selection.cross_validate(test_estimator, map.X_test, map.y_test,
                                                         scoring=score, cv=cv_)['test_score'].mean()
        history_note = [list(copy.deepcopy(add_features_keys)), score_value]
        history.append(history_note)
        print(history_note)
        if score_value > feature_max_score[1]:
            feature_max_score = (current_features_keys, score_value)
        if feature_max_score[1] - best_score > luft:
            best_score = feature_max_score[1]
            best_comp = current_features_keys
    history = list(np.array(history)[1:])
    if to_plot_history:
        history_score = np.array(history)[:, 1]
        plt.plot(range(len(history_score)), history_score)
        plt.show()
    return best_comp, best_score#, history

def brute_force_rf(samples_set, test_samples=None, uniform_samples=False, non_classified=False,
                   non_classified_color='black',
                   classes_borders=None, non_classified_random_points_number=1000,
                   potential_features=None, first_features=None, score='accuracy', cv_=10,
                   greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10,
                   n_estimators=(10, 20), max_depth=(3, 5), class_weight=('balanced', 'balanced_subsample'),
                   luft=0, to_plot_history=False):
    if potential_features is None:
        current_potential_features = list(samples_set.features_dict.keys())
    else:
        current_potential_features = potential_features
    if first_features is None:
        first_features = []
    best_comp = []
    best_score = 0.0
    feature_max_score = (None, 0.0)
    history = [[[], 0.0]]
    # вероятности для колличества признаков в комбинации
    for i in range(1, len(current_potential_features) + 1):
        for features_comb in itertools.combinations(current_potential_features, i):
            current_features_keys = first_features + list(features_comb)
            map = Map(samples_set, current_features_keys, uniform_samples=uniform_samples,
                      non_classified=non_classified, non_classified_color=non_classified_color,
                      classes_borders=classes_borders,
                      non_classified_random_points_number=non_classified_random_points_number)
            test_estimator = map.to_fit_by_RF(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                                              k_fold_order=k_fold_order, n_estimators=n_estimators,
                                              max_depth=max_depth, class_weight=class_weight)
            if test_samples is not None:
                score_value = map.to_test_mashine(test_sample_set=test_samples, features=current_features_keys,
                                                  uniform_samples=True)
            else:
                score_value = model_selection.cross_validate(test_estimator, map.X_test, map.y_test,
                                                             scoring=score, cv=cv_)['test_score'].mean()
            history_note = [list(copy.deepcopy(features_comb)), score_value]
            history.append(history_note)
            if score_value > feature_max_score[1]:
                feature_max_score = (current_features_keys, score_value)
            if feature_max_score[1] - best_score > luft:
                best_score = feature_max_score[1]
                best_comp = current_features_keys
    history = list(np.array(history)[1:])
    if to_plot_history:
        history_score = np.array(history)[:, 1]
        plt.plot(range(len(history_score)), history_score)
        plt.show()
    return best_comp, best_score#, history

def brute_force_knn(samples_set, test_samples=None, uniform_samples=False, non_classified=False,
                   non_classified_color='black',
                   classes_borders=None, non_classified_random_points_number=1000,
                   potential_features=None, first_features=None, score='accuracy', cv_=10,
                   greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10,
                    n_neighbors=(5, 10), weights=('uniform', 'distance'),
                    algorithm=('auto', 'ball_tree', 'kd_tree', 'brute'), p=(2, 1),
                   luft=0, to_plot_history=False):
    if potential_features is None:
        current_potential_features = list(samples_set.features_dict.keys())
    else:
        current_potential_features = potential_features
    if first_features is None:
        first_features = []
    best_comp = []
    best_score = 0.0
    feature_max_score = (None, 0.0)
    history = [[[], 0.0]]
    # вероятности для колличества признаков в комбинации
    for i in range(1, len(current_potential_features) + 1):
        for features_comb in itertools.combinations(current_potential_features, i):
            current_features_keys = first_features + list(features_comb)
            map = Map(samples_set, current_features_keys, uniform_samples=uniform_samples,
                      non_classified=non_classified, non_classified_color=non_classified_color,
                      classes_borders=classes_borders,
                      non_classified_random_points_number=non_classified_random_points_number)
            test_estimator = map.to_fit_by_KNN(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                                               k_fold_order=k_fold_order, n_neighbors=n_neighbors, weights=weights,
                                               algorithm=algorithm, p=p)
            if test_samples is not None:
                score_value = map.to_test_mashine(test_sample_set=test_samples, features=current_features_keys,
                                                  uniform_samples=True)
            else:
                score_value = model_selection.cross_validate(test_estimator, map.X_test, map.y_test,
                                                             scoring=score, cv=cv_)['test_score'].mean()
            history_note = [list(copy.deepcopy(features_comb)), score_value]
            history.append(history_note)
            if score_value > feature_max_score[1]:
                feature_max_score = (current_features_keys, score_value)
            if feature_max_score[1] - best_score > luft:
                best_score = feature_max_score[1]
                best_comp = current_features_keys
    history = list(np.array(history)[1:])
    if to_plot_history:
        history_score = np.array(history)[:, 1]
        plt.plot(range(len(history_score)), history_score)
        plt.show()
    return best_comp, best_score, history

def brute_force_qda(samples_set, test_samples=None, uniform_samples=False, non_classified=False,
                   non_classified_color='black',
                   classes_borders=None, non_classified_random_points_number=1000,
                   potential_features=None, first_features=None, score='accuracy', cv_=10,
                   greed_search=False, search_scoring='accuracy', cv=3, k_fold_order=10,
                   reg_param=(0, -0.1, 0.1), luft=0, to_plot_history=False):
    if potential_features is None:
        current_potential_features = list(samples_set.features_dict.keys())
    else:
        current_potential_features = potential_features
    if first_features is None:
        first_features = []
    best_comp = []
    best_score = 0.0
    feature_max_score = (None, 0.0)
    history = [[[], 0.0]]
    # вероятности для колличества признаков в комбинации
    for i in range(1, len(current_potential_features) + 1):
        for features_comb in itertools.combinations(current_potential_features, i):
            current_features_keys = first_features + list(features_comb)
            map = Map(samples_set, current_features_keys, uniform_samples=uniform_samples,
                      non_classified=non_classified, non_classified_color=non_classified_color,
                      classes_borders=classes_borders,
                      non_classified_random_points_number=non_classified_random_points_number)
            test_estimator = map.to_fit_by_QDA(greed_search=greed_search, search_scoring=search_scoring, cv=cv,
                                               k_fold_order=k_fold_order, reg_param=reg_param)
            if test_samples is not None:
                score_value = map.to_test_mashine(test_sample_set=test_samples, features=current_features_keys,
                                                  uniform_samples=True)
            else:
                score_value = model_selection.cross_validate(test_estimator, map.X_test, map.y_test,
                                                             scoring=score, cv=cv_)['test_score'].mean()
            history_note = [list(copy.deepcopy(features_comb)), score_value]
            history.append(history_note)
            if score_value > feature_max_score[1]:
                feature_max_score = (current_features_keys, score_value)
            if feature_max_score[1] - best_score > luft:
                best_score = feature_max_score[1]
                best_comp = current_features_keys
    history = list(np.array(history)[1:])
    if to_plot_history:
        history_score = np.array(history)[:, 1]
        plt.plot(range(len(history_score)), history_score)
        plt.show()
    return best_comp, best_score, history