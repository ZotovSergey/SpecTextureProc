import numpy as np
import DataSamples
import Processing

from sklearn import model_selection

if __name__ == "__main__":
    # samples_address = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
    #                   'Классификации/Классификация леса на лиственный и хвойный/Промежуточные результаты/samples2.file'
    # test_samples_address = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/Классификации/' \
    #                        'Классификация леса на лиственный и хвойный/Промежуточные результаты/test_samples_80.file'
    #
    # samples = DataSamples.to_load_samples_set(samples_address)#.to_align_classes()
    # test_samples = DataSamples.to_load_samples_set(test_samples_address)
    #
    # kfold = model_selection.StratifiedKFold(random_state=1, n_splits=10)
    # #print(Processing.add_del_svm(samples, potential_features=['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
    # #                             # 'Autocorrelation-0.0', 'ClusterProminence-0.0',
    # #                             # 'ClusterShade-0.0', 'Contrast-0.0', 'Correlation-0.0',
    # #                             # 'DiffEntropy-0.0', 'DiffVariance-0.0', 'Dissimilarity-0.0',
    # #                             # 'Energy-0.0', 'Entropy-0.0', 'Homogeneity-0.0',
    # #                             # 'Homogeneity2-0.0', 'InfMeasureCorr1-0.0',
    # #                             # 'InfMeasureCorr2-0.0', 'MaxProb-0.0', 'SumAverage-0.0',
    # #                             # 'SumEntropy-0.0', 'SumSquares-0.0', 'SumVariance-0.0'],
    # #                             first_features=None,
    # #                             # ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'], score='accuracy', cv_=kfold,
    # #                             greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
    # #                             iterations=1, del_algorithm=False))
    # spec_features = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    # text_features = ['Autocorrelation-0.0 dist_1',
    #                   'ClusterShade-0.0 dist_1', 'ClusterShade-135.0 dist_1',
    #                   'Contrast-0.0 dist_1', 'Contrast-135.0 dist_1',
    #                   #'Contrast-0.0 dist_3', 'Contrast-135.0 dist_3',
    #                   'Correlation-0.0 dist_1', 'Correlation-135.0 dist_1']
    #
    # print(Processing.brute_force_rf(samples, 30, test_samples, potential_features=spec_features,
    #                            #first_features=['swir1', 'swir2'],#spec_features,
    #                             score='accuracy', cv_=kfold,
    #                             greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
    #                             n_estimators=[50], max_depth=[20], class_weight=['balanced'],
    #                             luft=0, to_plot_history=True))

    samples_address = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
                      'Классификации/Классификация леса на лиственный и хвойный/Промежуточные результаты/samples_sel_s2.file'
    test_samples_address = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
                           'Классификации/Классификация леса на лиственный и хвойный/Промежуточные результаты/test_samples_sel_s2.file'

    samples = DataSamples.to_load_samples_set(samples_address)  # .to_align_classes()
    test_samples = DataSamples.to_load_samples_set(test_samples_address)


    kfold = model_selection.StratifiedKFold(random_state=1, n_splits=10)
    # print(Processing.add_del_svm(samples, potential_features=['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
    #                             # 'Autocorrelation-0.0', 'ClusterProminence-0.0',
    #                             # 'ClusterShade-0.0', 'Contrast-0.0', 'Correlation-0.0',
    #                             # 'DiffEntropy-0.0', 'DiffVariance-0.0', 'Dissimilarity-0.0',
    #                             # 'Energy-0.0', 'Entropy-0.0', 'Homogeneity-0.0',
    #                             # 'Homogeneity2-0.0', 'InfMeasureCorr1-0.0',
    #                             # 'InfMeasureCorr2-0.0', 'MaxProb-0.0', 'SumAverage-0.0',
    #                             # 'SumEntropy-0.0', 'SumSquares-0.0', 'SumVariance-0.0'],
    #                             first_features=None,
    #                             # ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'], score='accuracy', cv_=kfold,
    #                             greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
    #                             iterations=1, del_algorithm=False))

    #print(Processing.brute_force_rf(samples, test_samples, potential_features=['blue', 'green', 'red', 'nir', 'nir2'],
    #                                 score='accuracy', cv_=kfold,
    #                                 greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
    #                                 n_estimators=[50], max_depth=[15],
    #                                 luft=0, to_plot_history=True))
#
    # print(Processing.brute_force_rf(samples, test_samples, potential_features=['Autocorrelation-0.0 dist_1',
    #                                                                               'ClusterShade-0.0 dist_1', 'ClusterShade-135.0 dist_1',
    #                                                                               'Contrast-0.0 dist_1', 'Contrast-135.0 dist_1',
    #                                                                               'Correlation-0.0 dist_1', 'Correlation-135.0 dist_1'],
    #                                  first_features=['blue', 'green', 'red', 'nir', 'nir2'],
    #                                 score='accuracy', cv_=kfold,
    #                                 greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
    #                                 n_estimators=[50], max_depth=[15],
    #                                 luft=0, to_plot_history=True))

    # print(Processing.brute_force_knn(samples, test_samples, potential_features=['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
    #                                  # first_features=['blue', 'green', 'red', 'nir', 'nir2'],
    #                                  score='accuracy', cv_=kfold,
    #                                  greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
    #                                  n_neighbors=[75],
    #                                  luft=0, to_plot_history=True))

    print(Processing.brute_force_knn(samples, test_samples, potential_features=['Autocorrelation-0.0 dist_1',
                                                                               'ClusterShade-0.0 dist_1',
                                                                               'Contrast-0.0 dist_1',
                                                                               'Correlation-0.0 dist_1',
                                                                               'Correlation-135.0 dist_1'],
                                    first_features=['blue', 'swir1', 'swir2'],
                                    score='accuracy', cv_=kfold,
                                    greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
                                     n_neighbors=[75],
                                    luft=0, to_plot_history=True))

    # print(Processing.brute_force_qda(samples, test_samples, potential_features=['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
    #                                  # first_features=['green', 'red', 'nir', 'nir2'],
    #                                  score='accuracy', cv_=kfold,
    #                                  greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
    #                                  luft=0, to_plot_history=True))
