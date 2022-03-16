# Тестирование адаптивного порога
# import numpy as np
#
# import DataShot
# import DataSamples
# import Processing
#
# import shapefile
# import os
#
# if __name__ == "__main__":
#     spectral_list = ['blue', 'green', 'red', 'nir', 'nir2']
#     texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#     #texture_list = []
#     text_directions = [0]
#     win = 80
#     dist = [1]
#     samples_features = [
#                        'blue', 'green', 'red', 'nir', 'nir2',
#                        'Autocorrelation-0.0 dist_1 layer_0',
#                        'ClusterShade-0.0 dist_1 layer_0',
#                        'Contrast-0.0 dist_1 layer_0',
#                        'Correlation-0.0 dist_1 layer_0'
#                        ]
#
#     shot_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_WorldView_win_80_grad_100.file'
#     train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/train_selection_win_80_grad_50.file'
#     #test_samples_address = 'D:/Проекты/Бронницы/Векторные данные/Бронницы_wgs_ацо.shp'
#     result_directory = 'D:/Проекты/Бронницы/Результаты'
#
#     samples = DataSamples.to_load_samples_set(train_samples_address)
#     shot = DataShot.to_load_data_shot(shot_address)
#     #shot.remove_shadows(offset=1, keys_list=spectral_list)
#
#     map = Processing.Map(samples, features=samples_features,
#                         non_classified=False, non_classified_color='darkviolet', uniform_samples='Oversampling',
#                         normalize=True)
#     #map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', n_neighbors=[50])
#     #map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10)
#     map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#                                          n_estimators=[100], max_depth=[20], class_weight=['balanced'])
#
#     text_directions_dict = {0:{
#                            'Autocorrelation': {1: [0]},
#                            'ClusterShade': {1: [0]},
#                            'Contrast': {1: [0]},
#                            'Correlation': {1: [0]}
#                            }}
#     #text_directions_dict = {}
#
#     texture_adj_dir_dict = text_directions_dict
#     #for texture in texture_list:
#     #    texture_adj_dir_dict[texture] = {}
#     #for texture in texture_list:
#     #    for dir in dist:
#     #        texture_adj_dir_dict[texture].update({dir: text_directions})
#
#     #map.to_draw_classes('SDGL-90.0 dist_1 layer_3', 'Entropy-90.0 dist_1 layer_1',
#     #                    hyperspaces_borders=(100, 0, 0, 100), plane_step=0.01)
#
#     map.to_process(shot, spec_features=spectral_list, texture_features=texture_list,
#                   texture_adjacency_directions_dict=text_directions_dict)
#     map.to_save_map('res_rf_win_80_grad_50', result_directory)
#
#     map.to_test_mashine('test', result_directory)
#     #
#     test_samples_address = 'D:/Проекты/Бронницы/Векторные данные/Бронницы_wgs_ацо.shp'
#     # mean_shp_address = result_directory + '/p_mean_50.shp'
#     dominant_shp_address = result_directory + '/dominant.shp'
#     map.to_calc_dominant(test_samples_address, dominant_shp_address, 'DOM', shot,
#                         fields_to_copy=None,
#                         #possible_dom_classes=['C', 'D'],
#                         spec_features=spectral_list,
#                         texture_features=texture_list,
#                         texture_adjacency_directions_dict=texture_adj_dir_dict,
#                         )
#     # map.to_calc_mean(test_samples_address, mean_shp_address, 'MEAN', shot,
#     #                  fields_to_copy=None,
#     #                  texture_features=texture_list,
#     #                  texture_adjacency_directions_dict=texture_adj_dir_dict,
#     #                  borders=[0, 1], space_field_name='WEIGTH'
#     #                  )
#     map.to_compare_areas(dominant_shp_address, 'SPEC', dominant_shp_address, 'DOM', 'id', 'dominant_test', result_directory)
#     # print(map.to_compare_areas_err(mean_shp_address, 'SKAL1', mean_shp_address, 'MEAN', 'id', weights_col_name='WEIGTH'))


# #Проект Бронницы (WorldView)
# import numpy as np
#
# import DataShot
# import DataSamples
# import Processing
#
# import shapefile
# import os
#
# if __name__ == "__main__":
#     spectral_list = ['blue', 'green', 'yellow', 'red', 'nir', 'nir2']
#     #spectral_list = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
#     texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#     #texture_list = []
#     text_directions = [0]
#     win = 80
#     dist = [1]
#     samples_features = [
#                        'blue', 'green', 'yellow', 'red', 'nir', 'nir2'
#                        ]
#     #samples_features = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
#     mask_samples_features = [
#                             'blue', 'green', 'yellow', 'red', 'nir', 'nir2',
#                             'Autocorrelation-0.0 dist_1 layer_0',
#                             'ClusterShade-0.0 dist_1 layer_0',
#                             'Contrast-0.0 dist_1 layer_0',
#                             'Correlation-0.0 dist_1 layer_0'
#     ]
#
#     shot_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_WorldView_2_prior_knn.file'
#     train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/train_selection_win_80_grad_50.file'
#     mask_train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/mask_train_selection_win_80_grad_50.file'
#     mask2_train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/mask2_train_selection_win_80_grad_50.file'
#     test_samples_address = 'D:/Проекты/Бронницы/Векторные данные/Бронницы_wgs_ацо.shp'
#     result_directory = 'D:/Проекты/Бронницы/Результаты/Article results'
#
#     #shape_address = 'D:/Проекты/Бронницы/Векторные данные/train_selection_1.shp'
#     border_shape_address = 'D:/Проекты/Бронницы/Векторные данные/Border.shp'
#
#     shape_address = 'D:/Проекты/Бронницы/Векторные данные/train_selection_WorldView.shp'
#
#     # samples = DataSamples.to_load_samples_set(train_samples_address)
#     # mask_samples = DataSamples.to_load_samples_set(mask_train_samples_address)
#     samples = DataSamples.to_load_samples_set(train_samples_address)
#     test_samples = DataSamples.to_load_samples_set(train_samples_address)
#     mask_samples = DataSamples.to_load_samples_set(mask_train_samples_address)
#     mask2_samples = DataSamples.to_load_samples_set(mask2_train_samples_address)
#     shot = DataShot.to_load_data_shot(shot_address)
#     color_map = {
#                  'O': 'green',
#                  'P': 'cyan',
#                  'S': 'dodgerblue',
#                  'A': 'red',
#                  'B': 'm',
#                  'L': 'orange',
#                  'LE': 'brown',
#                  'LIN': 'magenta'
#                  }
#     #shot.to_save_image_as_geotiff(shot.get_prior_shot(color_map), shot.spec_geo_trans, shot.spec_projection_ref, 'prior', result_directory)
#     #shot.to_make_ndvi('red', 'nir')
#     shadow_mask = shot.take_shadows(offset=1, keys_list=spectral_list)
#
#     mask_map = Processing.Map(mask_samples, features=mask_samples_features,
#                               non_classified=False, non_classified_color='darkviolet', uniform_samples=3000,
#                               normalize=True)
#     mask_map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', cv=10, n_neighbors=50)
#     # mask_map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10)
#     # mask_map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#     #                       n_estimators=[100], max_depth=[20], class_weight=['balanced'])
#     # mask_map.to_fit_by_SVM(greed_search=False, search_scoring='accuracy', cv=10)
#
#     text_directions_dict = {0:{
#                            'Autocorrelation': {1: [0]},
#                            'ClusterShade': {1: [0]},
#                            'Contrast': {1: [0]},
#                            'Correlation': {1: [0]}
#                            }}
#
#     texture_adj_dir_dict = text_directions_dict
#
#     mask = mask_map.to_process_mask(shot, 'Forest', spec_features=spectral_list, texture_features=texture_list,
#                                     texture_adjacency_directions_dict=text_directions_dict)
#     mask_map.to_save_map('WorldView_first_knn_win_80_grad_50', result_directory)
#
#     shot.remove_by_mask(mask)
#
#     mask2_map = Processing.Map(mask2_samples, features=samples_features,
#                                non_classified=False, non_classified_color='darkviolet', uniform_samples='Oversampling',
#                                normalize=True)
#     mask2_map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', cv=10, n_neighbors=50)
#     # mask2_map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10)
#     # mask2_map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#     #                      n_estimators=[100], max_depth=[20], class_weight=['balanced'])
#     # mask2_map.to_fit_by_SVM(greed_search=False, search_scoring='accuracy', cv=10)
#
#     mask2 = mask2_map.to_process_mask(shot, 'Forest', spec_features=spectral_list, texture_features=[])
#     mask2_map.to_save_map('mask2_knn_win_80_grad_50', result_directory)
#
#     shot.remove_by_mask(mask2)
#     shot.remove_by_mask(shadow_mask)
#
#     #shot.to_save_data_shot('', intermediate_data_directory)
#
#     m_prob = [1, 2, 5, 10]
#     main_prob = {'A': {0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 0, 7: 0},
#                  'B': {0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 0, 7: 0},
#                  'L': {0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 0, 7: 0},
#                  'LE': {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0},
#                  'LIN': {0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 0, 7: 0},
#                  'O': {0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 0, 7: 0},
#                  'P': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1},
#                  'S': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1}}
#
#     map = Processing.Map(samples, features=samples_features,
#                          non_classified=False, non_classified_color='darkviolet', uniform_samples=3000,
#                          normalize=True)
#     map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', n_neighbors=50)
#     map.self_test('WorldView_test_knn', result_directory, cross_validation=True, reclassification=True,
#                   sample_remainder=False)
#
#     for m in m_prob:
#         for i, key in enumerate(main_prob.keys()):
#             main_prob[key][i] = m_prob
#         map = Processing.Map(samples, features=samples_features,
#                              non_classified=False, non_classified_color='darkviolet', uniform_samples=3000,
#                              normalize=True)
#         map.to_fit_dict_by_KNN(m, greed_search=False, search_scoring='accuracy', n_neighbors=50)
#         #map.to_fit_dict_by_KNN(100, 1, greed_search=False, search_scoring='accuracy', n_neighbors=50)
#         # map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10)
#         # map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#         #                 n_estimators=[100], max_depth=[20], class_weight=['balanced'])
#         # map.to_fit_dict_by_SVM(m, greed_search=False, search_scoring='accuracy', cv=10)
#
#         # map.to_test_mashine('test', result_directory)
#         # map.self_test('WorldView_test_' + str(m), result_directory, cross_validation=True, reclassification=True, sample_remainder=False)
#         #samples = DataSamples.to_load_samples_set(train_samples_address)
#         # map.to_test_mashine('test2', result_directory, features=samples_features, test_sample_set=samples)
#
#         # shot.remove_by_mask(mask2)
#         # shot.mask_by_polygons(shape_address)
#
#         # map = Processing.Map(samples, features=samples_features,
#         #                      non_classified=False, non_classified_color='darkviolet', uniform_samples='Oversampling',
#         #                      normalize=True)
#         #
#         # map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', n_neighbors=[50])
#         # #map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10)
#         # # map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#         # #                  n_estimators=[100], max_depth=[20], class_weight=['balanced'])
#         # #map.to_fit_by_SVM(greed_search=False, search_scoring='accuracy', cv=10)
#         #
#         # #map.to_test_mashine('test', result_directory)
#
#         map.to_process(shot, spec_features=spectral_list, texture_features=[], unbalanced_classifier=True)
#         map.to_save_map('WorldView_knn_map_' + str(m), result_directory)
#         for prior_class in color_map.keys():
#             map.maps_dict[prior_class].to_save_map(prior_class + '_knn_' + str(m), result_directory)
#
#         # mean_shp_address = result_directory + '/p_mean_50.shp'
#         dominant_shp_address = result_directory + '/WorldView_knn_dominant_' + str(m) + '.shp'
#         map.to_calc_dominant(test_samples_address, dominant_shp_address, 'DOM', shot,
#                             fields_to_copy=None,
#                             spec_features=spectral_list,
#                             texture_features=[],
#                             space_field_name='WEIGTH',
#                              unbalanced_classifier=True
#                             )
#         # map.to_calc_mean(test_samples_address, mean_shp_address, 'MEAN', shot,
#         #                  fields_to_copy=None,
#         #                  texture_features=texture_list,
#         #                  texture_adjacency_directions_dict=texture_adj_dir_dict,
#         #                  borders=[0, 1], space_field_name='WEIGTH'
#         #                  )
#         map.to_compare_areas(dominant_shp_address, 'SPEC', dominant_shp_address, 'DOM', 'id', 'knn_dominant_test' + str(m), result_directory)
#         # map.to_compare_areas(dominant_shp_address, 'SPEC', dominant_shp_address, 'DOM', 'id', 'knn_area_dominant_test', result_directory, weights_col_name='WEIGTH')
#         # print(map.to_compare_areas_err(mean_shp_address, 'SKAL1', mean_shp_address, 'MEAN', 'id', weights_col_name='WEIGTH'))
#
#         full_map = map.paint_by_mask(np.logical_not(shadow_mask), 'black').add_map(mask_map.paint_by_mask(mask, [0, 0, 0, 0])).add_map(mask2_map.paint_by_mask(mask2, [0, 0, 0, 0]))
#         full_map.to_save_map('WorldView_knn_full_map_' + str(m), result_directory)

# # Проект Бронницы (Resurs-P)
# import numpy as np
#
# import DataShot
# import DataSamples
# import Processing
#
# import shapefile
# import os
#
# if __name__ == "__main__":
#     spectral_list = ['blue', 'green', 'red', 'nir']
#     texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#     # texture_list = []
#     text_directions = [0]
#     win = 80
#     dist = [1]
#     samples_features = [
#         'blue', 'green', 'red', 'nir'
#     ]
#     #samples_features = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
#
#     mask_samples_features = [
#         'blue', 'green', 'red', 'nir',
#         'Autocorrelation-0.0 dist_1 layer_0',
#         'ClusterShade-0.0 dist_1 layer_0',
#         'Contrast-0.0 dist_1 layer_0',
#         'Correlation-0.0 dist_1 layer_0'
#     ]
#
#     shot_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_Resurs_win_80_grad_50_prior.file'
#     train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/train_selection_win_80_grad_50.file'
#     mask_train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/mask_train_selection_win_80_grad_50.file'
#     mask2_train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/mask2_train_selection_win_80_grad_50.file'
#     test_samples_address = 'D:/Проекты/Бронницы/Векторные данные/Бронницы_wgs_ацо_resurs.shp'
#     result_directory = 'D:/Проекты/Бронницы/Результаты/New sentinel 2'
#
#     shape_address = 'D:/Проекты/Бронницы/Векторные данные/train_selection_sentinel.shp'
#
#     samples = DataSamples.to_load_samples_set(train_samples_address)
#     test_samples = DataSamples.to_load_samples_set(train_samples_address)
#     mask_samples = DataSamples.to_load_samples_set(mask_train_samples_address)
#     mask2_samples = DataSamples.to_load_samples_set(mask2_train_samples_address)
#     shot = DataShot.to_load_data_shot(shot_address)
#     color_map = {
#                  'O': 'green',
#                  'P': 'cyan',
#                  'S': 'dodgerblue',
#                  'A': 'red',
#                  'B': 'm',
#                  'L': 'orange'
#                  }
#     #shot.to_make_pseudo_color_image('pseudo_color_image', result_directory, 'red', 'green', 'nir')
#     shot.remove_shadows(offset=1, keys_list=spectral_list)
#
#     mask_map = Processing.Map(mask_samples, features=mask_samples_features,
#                               non_classified=False, non_classified_color='darkviolet', uniform_samples='Oversampling',
#                               normalize=True)
#     # mask_map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', cv=10, n_neighbors=50, k_fold_order=None)
#     # mask_map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10)
#     # mask_map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#     #                      n_estimators=[100], max_depth=[20], class_weight=['balanced'])
#     mask_map.to_fit_by_SVM(greed_search=False, search_scoring='accuracy', cv=None, k_fold_order=None)
#
#     text_directions_dict = {0: {
#         'Autocorrelation': {1: [0]},
#         'ClusterShade': {1: [0]},
#         'Contrast': {1: [0]},
#         'Correlation': {1: [0]}
#     }}
#
#     texture_adj_dir_dict = text_directions_dict
#
#     mask_classes = ['Forest']
#
#     mask = mask_map.to_process_mask(shot, mask_classes, spec_features=spectral_list, texture_features=texture_list,
#                                     texture_adjacency_directions_dict=text_directions_dict)
#     mask_map.to_save_map('mask_svm_win_80_grad_50', result_directory)
#     shot.remove_by_mask(mask)
#
#     mask2_map = Processing.Map(mask2_samples, features=samples_features,
#                               non_classified=False, non_classified_color='darkviolet', uniform_samples='Oversampling',
#                               normalize=True)
#     # mask2_map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', cv=10, n_neighbors=50, k_fold_order=None)
#     # mask2_map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10)
#     # mask2_map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#     #                      n_estimators=[100], max_depth=[20], class_weight=['balanced'])
#     mask2_map.to_fit_by_SVM(greed_search=False, search_scoring='accuracy', cv=None, k_fold_order=None)
#
#     mask2 = mask2_map.to_process_mask(shot, mask_classes, spec_features=spectral_list, texture_features=[])
#     mask2_map.to_save_map('mask2_svm_win_80_grad_50', result_directory)
#
#     map = Processing.Map(samples, features=samples_features,
#                          non_classified=False, non_classified_color='darkviolet', uniform_samples=6000,
#                          normalize=True)
#     m_prob = 10
#     main_prob = {'A': {0: 1, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0},
#                  'B': {0: 1, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0},
#                  'L': {0: 1, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0},
#                  'O': {0: 1, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0},
#                  'P': {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1},
#                  'S': {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1}}
#     for i, key in enumerate(main_prob.keys()):
#         main_prob[key][i] = m_prob
#     # map.to_fit_dict_by_KNN(main_prob, greed_search=False, search_scoring='accuracy', n_neighbors=50, k_fold_order=None)
#     # map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10)
#     # map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#     #                 n_estimators=[100], max_depth=[20], class_weight=['balanced'])
#     map.to_fit_dict_by_SVM(main_prob, greed_search=False, search_scoring='accuracy', cv=None, k_fold_order=None)
#
#     # map.to_test_mashine('test', result_directory)
#     map.self_test('test', result_directory, cross_validation=True, reclassification=True, sample_remainder=False)
#
#     shot.remove_by_mask(mask2)
#     #shot.mask_by_polygons(shape_address)
#
#     map.to_process(shot, spec_features=spectral_list, texture_features=[], unbalanced_classifier=True)
#     map.to_test_estimator_by_polygons(shape_address, 'CLASS', 'prior_test', result_directory)
#     map.to_save_map('svm_map', result_directory)
#     for prior_class in color_map.keys():
#         map.maps_dict[prior_class].to_save_map(prior_class, result_directory)
#
#     # mean_shp_address = result_directory + '/p_mean_50.shp'
#     dominant_shp_address = result_directory + '/svm_dominant.shp'
#     map.to_calc_dominant(test_samples_address, dominant_shp_address, 'DOM', shot,
#                         fields_to_copy=None,
#                         spec_features=spectral_list,
#                         texture_features=[],
#                         space_field_name='WEIGTH',
#                          unbalanced_classifier=True
#                         )
#     # # map.to_calc_mean(test_samples_address, mean_shp_address, 'MEAN', shot,
#     # #                  fields_to_copy=None,
#     # #                  texture_features=texture_list,
#     # #                  texture_adjacency_directions_dict=texture_adj_dir_dict,
#     # #                  borders=[0, 1], space_field_name='WEIGTH'
#     # #                  )
#     map.to_compare_areas(dominant_shp_address, 'SPEC', dominant_shp_address, 'DOM', 'id', 'svm_dominant_test',
#                          result_directory)
#     # map.to_compare_areas(dominant_shp_address, 'SPEC', dominant_shp_address, 'DOM', 'id', 'knn_area_dominant_test',
#     #                      result_directory, weights_col_name='WEIGTH')
#     # # print(map.to_compare_areas_err(mean_shp_address, 'SKAL1', mean_shp_address, 'MEAN', 'id', weights_col_name='WEIGTH'))

# # Проект Бронницы (Sentinel simp)
# import numpy as np
#
# import DataShot
# import DataSamples
# import Processing
#
# import shapefile
# import os
#
# if __name__ == "__main__":
#     prefixes = [
#                 '(2021.09.13)',
#                 '(2021.07.20)',
#                 '(2021.07.18)',
#                 '(2021.07.13)',
#                 '(2021.06.23)',
#                 '(2021.06.20)',
#                 '(2021.06.18)',
#                 '(2021.06.03)',
#                 '(2020.10.28)',
#                 '(2020.10.01)',
#                 '(2020.09.26)',
#                 '(2020.09.25)',
#                 '(2020.09.06)',
#                 '(2020.06.10)',
#                 '(2019.08.20)',
#                 '(2019.06.06)',
#                 '(2019.06.04)',
#                 '(2018.10.17)',
#                 '(2018.10.14)',
#                 '(2018.10.09)',
#                 '(2018.09.22)',
#                 '(2018.09.02)',
#                 '(2018.08.25)',
#                 '(2018.08.10)',
#                 '(2018.05.25)',
#                 '(2018.05.12)',
#                 '(2018.05.10)',
#                 '(2017.09.24)',
#                 '(2017.07.29)',
#                 '(2017.05.07)',
#                 '(2016.09.06)',
#                 '(2016.07.24)',
#                 '(2015.09.18)',
#                 '(2015.08.09)',
#                 '(2020.11.17)',
#                 '(2019.11.23)',
#                 '(2018.11.23)',
#                 '(2018.11.16)',
#                 '(2018.11.06)',
#                 '(2017.11.08)'
#                 ]
#
#
#     for prefix in prefixes:
#         spectral_list = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
#         texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#         text_directions = [0]
#         win = 80
#         dist = [1]
#         samples_features = [
#             'blue', 'green', 'red', 'nir', 'swir1', 'swir2'
#         ]
#
#         mask_samples_features = [
#             'blue', 'green', 'red', 'nir', 'swir1', 'swir2',
#             'Autocorrelation-0.0 dist_1 layer_0',
#             'ClusterShade-0.0 dist_1 layer_0',
#             'Contrast-0.0 dist_1 layer_0',
#             'Correlation-0.0 dist_1 layer_0'
#         ]
#
#         shot_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_Sentinel_win_80_grad_50_' + prefix + '.file'
#         train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/train_selection_win_80_grad_50 D ' + prefix + '.file'
#         mask_train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/mask_train_selection_win_80_grad_50 ' + prefix + '.file'
#         mask2_train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/mask2_train_selection_win_80_grad_50 ' + prefix + '.file'
#         test_samples_address = 'D:/Проекты/Бронницы/Векторные данные/Бронницы_wgs_ацо_resurs.shp'
#         intermediate_data_directory = 'D:/Проекты/Бронницы/Промежуточные данные'
#         result_directory = 'D:/Проекты/Бронницы/Результаты/graph/Sentinel knn D 40 ' + prefix
#
#         #shape_address = 'D:/Проекты/Бронницы/Векторные данные/train_selection_resurs.shp'
#
#         samples = DataSamples.to_load_samples_set(train_samples_address)
#         test_samples = DataSamples.to_load_samples_set(train_samples_address)
#         mask_samples = DataSamples.to_load_samples_set(mask_train_samples_address)
#         mask2_samples = DataSamples.to_load_samples_set(mask2_train_samples_address)
#         shot = DataShot.to_load_data_shot(shot_address)
#
#         mask_map = Processing.Map(mask_samples, features=mask_samples_features,
#                                   non_classified=False, non_classified_color='darkviolet', uniform_samples='Oversampling',
#                                   normalize=True)
#         mask_map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', cv=10, n_neighbors=50)
#         # mask_map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10)
#         # mask_map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#         #                      n_estimators=[100], max_depth=[20], class_weight=['balanced'])
#         # mask_map.to_fit_by_SVM(greed_search=False, search_scoring='accuracy', cv=10, decision_function_shape='ecoc')
#
#         text_directions_dict = {0: {
#             'Autocorrelation': {1: [0]},
#             'ClusterShade': {1: [0]},
#             'Contrast': {1: [0]},
#             'Correlation': {1: [0]}
#         }}
#
#         texture_adj_dir_dict = text_directions_dict
#
#         mask_classes = ['Forest']
#
#         mask = mask_map.to_process_mask(shot, mask_classes, spec_features=spectral_list, texture_features=texture_list,
#                                         texture_adjacency_directions_dict=text_directions_dict)
#         mask_map.to_save_map('mask_knn_win_80_grad_50', result_directory)
#         shot.remove_by_mask(mask)
#
#         mask2_map = Processing.Map(mask2_samples, features=samples_features,
#                                   non_classified=False, non_classified_color='darkviolet', uniform_samples='Oversampling',
#                                   normalize=True)
#         mask2_map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', cv=10, n_neighbors=50)
#         # mask2_map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10)
#         # mask2_map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#         #                      n_estimators=[100], max_depth=[20], class_weight=['balanced'])
#         # mask2_map.to_fit_by_SVM(greed_search=False, search_scoring='accuracy', cv=10, decision_function_shape='ecoc')
#
#         mask2 = mask2_map.to_process_mask(shot, mask_classes, spec_features=spectral_list, texture_features=[])
#         mask2_map.to_save_map('mask2_knn_win_80_grad_50', result_directory)
#
#         shot.remove_by_mask(mask2)
#
#         shot.remove_by_mask(mask)
#         shot.remove_by_mask(mask2)
#
#         map = Processing.Map(samples, features=samples_features,
#                                non_classified=False, non_classified_color='darkviolet', uniform_samples='Oversampling',
#                                normalize=True)
#         map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', n_neighbors=40)
#         # map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10, priors=class_weights)
#         # map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#         #                    n_estimators=100, max_depth=20, class_weight=class_weights)
#         # map.to_fit_by_SVM(greed_search=False, search_scoring='accuracy', cv=10, decision_function_shape='ecoc')
#
#         #map.to_test_mashine('test', result_directory)
#
#         map.self_test('test', result_directory, cross_validation=True, reclassification=True, sample_remainder=False)
#
#         map.to_process(shot, spec_features=spectral_list, texture_features=[])
#         map.to_save_map('knn_map', result_directory)
#
#         dominant_shp_address = result_directory + '/knn_dominant.shp'
#         map.to_calc_dominant(test_samples_address, dominant_shp_address, 'DOM', shot,
#                              fields_to_copy=None,
#                              spec_features=spectral_list,
#                              texture_features=[],
#                              space_field_name='WEIGTH'
#                              )
#
#         map.to_compare_areas(dominant_shp_address, 'SPEC', dominant_shp_address, 'DOM', 'id', 'knn_dominant_test',
#                              result_directory)

# # Проект Бронницы (Sentinel)
# import numpy as np
#
# import DataShot
# import DataSamples
# import Processing
#
# import shapefile
# import os
#
# if __name__ == "__main__":
#     spectral_list = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
#     texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#     text_directions = [0]
#     win = 80
#     dist = [1]
#     samples_features = [
#         'blue', 'green', 'red', 'nir', 'swir1', 'swir2'
#     ]
#
#     new_samples_features = [
#         'green', 'red', 'nir', 'swir1', 'swir2'
#     ]
#
#     mask_samples_features = [
#         'blue', 'green', 'red', 'nir', 'swir1', 'swir2',
#         'Autocorrelation-0.0 dist_1 layer_0',
#         'ClusterShade-0.0 dist_1 layer_0',
#         'Contrast-0.0 dist_1 layer_0',
#         'Correlation-0.0 dist_1 layer_0'
#     ]
#
#     WorldView_shot_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_WorldView_win_80_grad_50.file'
#     # Resurs_shot_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_Resurs_win_80_grad_50.file'
#     mask_shot_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_Sentinel_win_80_grad_50_full_(2021.07.18).file'
#     DC_shot_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_Sentinel_win_80_grad_50_full_(2018.11.23).file'
#     DLE_shot_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_Sentinel_win_80_grad_50_full_(2018.10.14).file'
#     C_shot_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_Sentinel_win_80_grad_50_full_(2021.09.13).file'
#     D2_shot_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_Sentinel_win_80_grad_50_full_(2017.07.29).file'
#     D22_shot_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_Sentinel_win_80_grad_50_full_(2016.07.24).file'
#     mask_train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/mask_train_selection_win_80_grad_50.file'
#     # mask2_train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/mask2_train_selection_win_80_grad_50.file'
#     train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/train_selection_win_80_grad_50.file'
#     DC_train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/DC_train_selection_win_80_grad_50.file'
#     DLE_train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/DLE_train_selection_win_80_grad_50.file'
#     LE_train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/LE_train_selection_win_80_grad_50.file'
#     C_train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/C_train_selection_win_80_grad_50.file'
#     D2_train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/D2_train_selection_win_80_grad_50.file'
#     D22_train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/D22_train_selection_win_80_grad_50.file'
#     test_shape_address = 'D:/Проекты/Бронницы/Векторные данные/Бронницы_wgs_ацо.shp'
#     intermediate_data_directory = 'D:/Проекты/Бронницы/Промежуточные данные'
#     result_directory = 'D:/Проекты/Бронницы/Результаты/Test calc'
#
#     mask_samples = DataSamples.to_load_samples_set(mask_train_samples_address)
#     DC_samples = DataSamples.to_load_samples_set(DC_train_samples_address)
#     DLE_samples = DataSamples.to_load_samples_set(DLE_train_samples_address)
#     LE_samples = DataSamples.to_load_samples_set(LE_train_samples_address)
#     C_samples = DataSamples.to_load_samples_set(C_train_samples_address)
#     D2_samples = DataSamples.to_load_samples_set(D2_train_samples_address)
#     D22_samples = DataSamples.to_load_samples_set(D22_train_samples_address)
#     samples = DataSamples.to_load_samples_set(train_samples_address)
#     mask_shot = DataShot.to_load_data_shot(mask_shot_address)
#     DC_shot = DataShot.to_load_data_shot(DC_shot_address)
#     DLE_shot = DataShot.to_load_data_shot(DLE_shot_address)
#     C_shot = DataShot.to_load_data_shot(C_shot_address)
#     D2_shot = DataShot.to_load_data_shot(D2_shot_address)
#     D22_shot = DataShot.to_load_data_shot(D22_shot_address)
#     ww_shot = DataShot.to_load_data_shot(WorldView_shot_address)
#     #rp_shot = DataShot.to_load_data_shot(Resurs_shot_address)
#
#     color_map = {
#         'O': 'green',
#         'A': 'red',
#         'B': 'blueviolet',
#         'L': 'orange',
#         'LIN': 'magenta'
#     }
#     D2_shape_address = 'D:/Проекты/Бронницы/Векторные данные/train_selection_sentinel_D2.shp'
#     D2_shot.transform_to_other_shot(D22_shot, D2_shape_address, spec_bands_keys=samples_features, class_field='CLASS',
#                                     reg_color='blue', color_map=color_map, class_mean=True)
#
#     # D2_samples.mean_spector(channels=['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#     #                         classes=['A', 'B', 'L', 'LIN', 'O'],
#     #                         graph_colors = ['red', 'blueviolet', 'orange', 'magenta', 'green'])
#     # D22_samples.mean_spector(channels=['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#     #                         classes=['A', 'B', 'L', 'LIN', 'O'],
#     #                         graph_colors=['red', 'blueviolet', 'orange', 'magenta', 'green'])
#
#     mask_map = Processing.Map(mask_samples, features=mask_samples_features,
#                               non_classified=False, non_classified_color='darkviolet', uniform_samples='Oversampling',
#                               normalize=True)
#     mask_map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', cv=10, n_neighbors=40)
#     # mask_map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10)
#     # mask_map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#     #                      n_estimators=[100], max_depth=[20], class_weight=['balanced'])
#     # mask_map.to_fit_by_SVM(greed_search=False, search_scoring='accuracy', cv=10)
#
#     text_directions_dict = {0: {
#         'Autocorrelation': {1: [0]},
#         'ClusterShade': {1: [0]},
#         'Contrast': {1: [0]},
#         'Correlation': {1: [0]}
#     }}
#
#     texture_adj_dir_dict = text_directions_dict
#
#     mask = mask_map.to_process_mask(mask_shot, 'Forest', spec_features=spectral_list, texture_features=texture_list,
#                                     texture_adjacency_directions_dict=text_directions_dict)
#
#     mask_map.to_save_map('mask_knn_win_80_grad_50', result_directory)
#     DC_shot.remove_by_mask(mask)
#     DLE_shot.remove_by_mask(mask)
#     C_shot.remove_by_mask(mask)
#     D2_shot.remove_by_mask(mask)
#     D22_shot.remove_by_mask(mask)
#
#
#     DC_map = Processing.Map(DC_samples, features=samples_features,
#                          non_classified=False, non_classified_color='darkviolet', uniform_samples='Oversampling',
#                          normalize=True)
#
#     DC_map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', n_neighbors=40)
#     # DC_map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10)
#     # DC_map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#     #                 n_estimators=[100], max_depth=[20], class_weight=['balanced'])
#     # DC_map.to_fit_by_SVM(greed_search=False, search_scoring='accuracy', cv=10)
#
#     DC_map.self_test('DC_knn_test', result_directory, cross_validation=True, reclassification=True, sample_remainder=False)
#
#     D_mask = DC_map.to_process_mask(DC_shot, ['O', 'A', 'B', 'L', 'LIN', 'LE'], spec_features=spectral_list, texture_features=[])
#     C_mask = DC_map.to_process_mask(DC_shot, ['P', 'S'], spec_features=spectral_list, texture_features=[])
#     DC_map.to_save_map('DC_knn_map', result_directory)
#
#     right_class_percent_shp_address = result_directory + '/correct_type_percent_knn.shp'
#     DC_map.to_change_color_map(classes=['D', 'C'], color_map=['green', 'cyan'])
#     DC_map.to_calc_right_percent(test_shape_address, right_class_percent_shp_address, 'CTP', 'TYPE')
#
#     DLE_shot.remove_by_mask(D_mask)
#     C_shot.remove_by_mask(C_mask)
#
#     DLE_map = Processing.Map(DLE_samples, features=samples_features,
#                            non_classified=False, non_classified_color='darkviolet', uniform_samples='Oversampling',
#                            normalize=True)
#     DLE_map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', n_neighbors=40)
#     # DLE_map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10, priors=class_weights)
#     # DLE_map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#     #                    n_estimators=100, max_depth=20, class_weight=class_weights)
#     # DLE_map.to_fit_by_SVM(greed_search=False, search_scoring='accuracy', cv=10)
#
#     DLE_map.self_test('DLE_test_knn', result_directory, cross_validation=True, reclassification=True, sample_remainder=False)
#
#     DLE_map.to_process(DLE_shot, spec_features=spectral_list, texture_features=[])
#
#     DLE_map.to_save_map('DLE_knn_map', result_directory)
#
#     D2_mask = DLE_map.to_process_mask(DLE_shot, ['O', 'A', 'B', 'L', 'LIN'], spec_features=spectral_list,
#                                     texture_features=[])
#     LE_mask = DLE_map.to_process_mask(DLE_shot, ['LE'], spec_features=spectral_list, texture_features=[])
#     D2_shot.remove_by_mask(D2_mask)
#     D22_shot.remove_by_mask(D2_mask)
#
#     LE_map = Processing.Map(LE_samples, features=samples_features,
#                             non_classified=False, non_classified_color='darkviolet', uniform_samples='Oversampling',
#                             normalize=True)
#     LE_map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', n_neighbors=40)
#
#     LE_shot = DLE_shot
#     LE_shot.remove_by_mask(LE_mask)
#     LE_map.to_process(LE_shot, spec_features=spectral_list, texture_features=[])
#
#     ww_shot.to_add_prior_classes(LE_map)
#     # rp_shot.to_add_prior_classes(LE_map)
#
#     C_map = Processing.Map(C_samples, features=samples_features,
#                            non_classified=False, non_classified_color='darkviolet', uniform_samples='Oversampling',
#                            normalize=True)
#     C_map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', n_neighbors=40)
#     # C_map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10, priors=class_weights)
#     # C_map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#     #                    n_estimators=100, max_depth=20, class_weight=class_weights)
#     # C_map.to_fit_by_SVM(greed_search=False, search_scoring='accuracy', cv=10)
#
#     C_map.self_test('C_test_knn', result_directory, cross_validation=True, reclassification=True, sample_remainder=False)
#     C_map.to_process(C_shot, spec_features=spectral_list, texture_features=[])
#
#     ww_shot.to_add_prior_classes(C_map)
#     # rp_shot.to_add_prior_classes(C_map)
#     C_map.to_save_map('C_knn_map', result_directory)
#
#     D2_map = Processing.Map(D2_samples, features=samples_features,
#                            non_classified=False, non_classified_color='darkviolet', uniform_samples='Oversampling',
#                            normalize=True)
#     D2_map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', n_neighbors=40)
#     # D2_map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10, priors=class_weights)
#     # D2_map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#     #                    n_estimators=100, max_depth=20, class_weight=class_weights)
#     # D2_map.to_fit_by_SVM(greed_search=False, search_scoring='accuracy', cv=10)
#
#     D2_map.self_test('D2_test_knn', result_directory, cross_validation=True, reclassification=True, sample_remainder=False)
#
#     D2_map.to_process(D22_shot, spec_features=samples_features, texture_features=[])
#     ww_shot.to_add_prior_classes(D2_map)
#     # rp_shot.to_add_prior_classes(D2_map)
#     D2_map.to_save_map('D2_knn_map', result_directory)
#
#     ww_shot.to_save_data_shot('Shot_WorldView_2_prior_knn', intermediate_data_directory)
#     # rp_shot.to_save_data_shot('Shot_Resurs_win_80_grad_50_prior', intermediate_data_directory)
#
#     map = LE_map.add_map(C_map).add_map(D2_map)
#     map.to_save_map('knn_map', result_directory)
#
#     i = 3
#     map.map[np.where((map.map[:, :, 0] == map.color_map[i][0]) &
#                      (map.map[:, :, 1] == map.color_map[i][1]) &
#                      (map.map[:, :, 2] == map.color_map[i][2]) &
#                      (map.map[:, :, 3] == map.color_map[i][3]))] = map.color_map[2]
#
#     dominant_shp_address = result_directory + '/knn_dominant.shp'
#     dominant2_shp_address = result_directory + '/knn_dominant2.shp'
#
#     # map.to_calc_dominant_2(test_shape_address, dominant_shp_address, 'DOM',
#     #                        fields_to_copy=None,
#     #                        space_field_name='WEIGTH', min_space=50,
#     #                        )
#     # map.to_calc_dominant_2(test_shape_address, dominant2_shp_address, 'DOM',
#     #                      fields_to_copy=None,
#     #                      space_field_name='WEIGTH',
#     #                      dom_order=2, min_space=10
#     #                      )
#
#     # map.to_compare_areas(dominant_shp_address, 'SPEC', dominant_shp_address, 'DOM', 'id', 'knn_dominant_test',
#     #                        result_directory)
#
#     right_class_percent_shp_address = result_directory + '/correct_class_percent_knn.shp'
#
#     map.to_calc_right_percent(test_shape_address, right_class_percent_shp_address, 'CCP', 'SPEC')
#
#     full_map = mask_map.add_map(LE_map.add_map(C_map).add_map(D2_map))
#     full_map.to_save_map('knn_full_map', result_directory)

# # Проект Бронницы (correction Sentinel)
# import numpy as np
#
# import DataShot
# import DataSamples
# import Processing
#
# import shapefile
# import os
#
# from matplotlib import pyplot as plt
#
# if __name__ == "__main__":
#     spectral_list = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
#     texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#     text_directions = [0]
#     win = 80
#     dist = [1]
#     model = -1
#     samples_features = [
#         'blue', 'green', 'red', 'nir', 'swir1', 'swir2'
#     ]
#
#     mask_samples_features = [
#         'blue', 'green', 'red', 'nir', 'swir1', 'swir2',
#         'Autocorrelation-0.0 dist_1 layer_0',
#         'ClusterShade-0.0 dist_1 layer_0',
#         'Contrast-0.0 dist_1 layer_0',
#         'Correlation-0.0 dist_1 layer_0'
#     ]
#
#     text_directions_dict = {0: {
#         'Autocorrelation': {1: [0]},
#         'ClusterShade': {1: [0]},
#         'Contrast': {1: [0]},
#         'Correlation': {1: [0]}
#     }}
#     texture_adj_dir_dict = text_directions_dict
#
#     WorldView_shot_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_WorldView_win_80_grad_50.file'
#     F1_shot_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_Sentinel_win_80_grad_50_full_aerosol_(2019.06.04).file'
#     F2_shot_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_Sentinel_win_80_grad_50_full_aerosol_(2019.06.06).file'
#     F1_shot_sen2corr_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_Sentinel_win_80_grad_50_full_corr_(2019.06.04).file'
#     F2_shot_sen2corr_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_Sentinel_win_80_grad_50_full_corr_(2019.06.06).file'
#     mask1_train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/mask1_train_selection_win_80_grad_50.file'
#     mask2_train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/mask2_train_selection_win_80_grad_50.file'
#     F1_train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/F1_train_selection_win_80_grad_50.file'
#     F2_train_samples_address = 'D:/Проекты/Бронницы/Промежуточные данные/F2_train_selection_win_80_grad_50.file'
#     test_shape_address = 'D:/Проекты/Бронницы/Векторные данные/Бронницы_wgs_ацо.shp'
#     intermediate_data_directory = 'D:/Проекты/Бронницы/Промежуточные данные'
#     result_directory = 'D:/Проекты/Бронницы/Результаты/Colibration calc'
#
#     mask1_samples = DataSamples.to_load_samples_set(mask1_train_samples_address)
#     mask2_samples = DataSamples.to_load_samples_set(mask2_train_samples_address)
#     F1_samples = DataSamples.to_load_samples_set(F1_train_samples_address)
#     F2_samples = DataSamples.to_load_samples_set(F2_train_samples_address)
#     F1_shot = DataShot.to_load_data_shot(F1_shot_address)
#     F2_shot = DataShot.to_load_data_shot(F2_shot_address)
#     F1_DOS_shot = DataShot.to_load_data_shot(F1_shot_address)
#     F2_DOS_shot = DataShot.to_load_data_shot(F2_shot_address)
#     F1_DOS_shot.DOS_colebration('aerosol', model, start_point='hist', show_correction_curve=False, show_start_point=False, map_band='red', neg_rule='calib')
#     F2_DOS_shot.DOS_colebration('aerosol', model, start_point='hist', show_correction_curve=False, show_start_point=False, map_band='red', neg_rule='calib')
#     # F1_DOS_shot.simple_DOS_colebration(start_point=1, show_correction_curve=True)
#     # F2_DOS_shot.simple_DOS_colebration(start_point=1, show_correction_curve=False)
#     F1_sen2corr_shot = DataShot.to_load_data_shot(F1_shot_sen2corr_address)
#     F2_sen2corr_shot = DataShot.to_load_data_shot(F2_shot_sen2corr_address)
#     F1_DOS_shot.name = 'corr_' + F1_DOS_shot.name
#     F2_DOS_shot.name = 'corr_' + F2_DOS_shot.name
#
#     color_map = {
#         'O': 'green',
#         'A': 'red',
#         'B': 'blueviolet',
#         'L': 'orange',
#         'LE': 'brown',
#         'LIN': 'magenta',
#         'P': 'cyan',
#         'S': 'dodgerblue'
#     }
#
#     F_shape_address = 'D:/Проекты/Бронницы/Векторные данные/train_selection_sentinel.shp'
#     x, y_F1 = F1_shot.graph_spectors(F_shape_address, spec_bands_keys=samples_features, class_field='CLASS', color_map=color_map, fig_num=1, name='04.06.2019')
#     x, y_F2 = F2_shot.graph_spectors(F_shape_address, spec_bands_keys=samples_features, class_field='CLASS', color_map=color_map, fig_num=2, name='06.06.2019')
#     # x, y_F1_corr = F1_DOS_shot.graph_spectors(F_shape_address, spec_bands_keys=samples_features, class_field='CLASS', color_map=color_map, fig_num=3, name='Коррекция DOS 04.06.2019')
#     # x, y_F2_corr = F2_DOS_shot.graph_spectors(F_shape_address, spec_bands_keys=samples_features, class_field='CLASS', color_map=color_map, fig_num=4, name='Коррекция DOS 06.06.2019')
#     x, y_F1_corr = F1_sen2corr_shot.graph_spectors(F_shape_address, spec_bands_keys=samples_features, class_field='CLASS', color_map=color_map, fig_num=3, name='Коррекция sen2corr 04.06.2019')
#     x, y_F2_corr = F2_sen2corr_shot.graph_spectors(F_shape_address, spec_bands_keys=samples_features, class_field='CLASS', color_map=color_map, fig_num=4, name='Коррекция sen2corr 06.06.2019')
#     F1_shot.graph_compare(F1_DOS_shot, F_shape_address, spec_bands_keys=samples_features, class_field='CLASS',
#                          reg_color='red', color_map=color_map, class_mean=False)
#     F2_shot.graph_compare(F2_DOS_shot, F_shape_address, spec_bands_keys=samples_features, class_field='CLASS',
#                          reg_color='blue', color_map=color_map, class_mean=False, legend=False)
#     F1_ndvi = (y_F1[:, 3] - y_F1[:, 2]) / (y_F1[:, 3] + y_F1[:, 2])
#     F2_ndvi = (y_F2[:, 3] - y_F2[:, 2]) / (y_F2[:, 3] + y_F2[:, 2])
#     F1_corr_ndvi = (y_F1_corr[:, 3] - y_F1_corr[:, 2]) / (y_F1_corr[:, 3] + y_F1_corr[:, 2])
#     F2_corr_ndvi = (y_F2_corr[:, 3] - y_F2_corr[:, 2]) / (y_F2_corr[:, 3] + y_F2_corr[:, 2])
#
#
#     class_set = ['A', 'B', 'L', 'LE', 'LIN', 'O', 'P', 'S']
#     y_F1_F1_corr = y_F1 - y_F1_corr
#     for i, clas in enumerate(class_set):
#         plt.plot(x, y_F1_F1_corr[i], color_map[clas], marker='o')
#     plt.legend(class_set)
#     plt.title('Разница после коррекции DOS (04.06.2019)')
#     plt.xlabel('Wavelength')
#     plt.grid()
#     plt.show()
#
#     y_F1_corr_F2_corr = y_F1_corr - y_F2_corr
#     for i, clas in enumerate(class_set):
#         plt.plot(x, y_F1_corr_F2_corr[i], color_map[clas], marker='o')
#     plt.legend(class_set)
#     plt.title('Разница между снимками 04.06.2019 и 06.06.2019 после коррекции DOS')
#     plt.xlabel('Wavelength')
#     plt.grid()
#     plt.show()
#
#     y_F1_F2_diff_F1_corr_F2_corr = (y_F1_corr - y_F2_corr) - (y_F1 - y_F2)
#     for i, clas in enumerate(class_set):
#         plt.plot(x, y_F1_F2_diff_F1_corr_F2_corr[i], color_map[clas], marker='o')
#     plt.legend(class_set)
#     plt.title('Уменьшение разницы между снимками 04.06.2019 и 06.06.2019 после коррекции DOS')
#     plt.xlabel('Wavelength')
#     plt.grid()
#     plt.show()
#
#     width = 0.4
#     x = np.arange(len(class_set))
#     F1_F2_ndvi = F1_ndvi - F2_ndvi
#     F1_corr_F2_corr_ndvi = F1_corr_ndvi - F2_corr_ndvi
#     plt.bar(x - 0.2, F1_F2_ndvi, width=width)
#     plt.bar(x + 0.2, F1_corr_F2_corr_ndvi, width=width)
#     plt.xticks(x, class_set)
#     plt.title('Разница NDVI между снимками 04.06.2019 и 06.06.2019')
#     plt.xlabel('Wavelength')
#     plt.legend(['До коррекции simple sen2corr', 'После коррекции simple sen2corr'])
#     plt.grid()
#     plt.show()
#
#     F1_F1_corr_ndvi = F1_ndvi - F1_corr_ndvi
#     plt.bar(class_set, F1_F1_corr_ndvi)
#     plt.title('Разница NDVI после коррекции sen2corr (04.06.2019)')
#     plt.xlabel('Wavelength')
#     plt.grid()
#     plt.show()
#
#     y_F1_F2_diff_F1_corr_F2_corr_ndvi = abs((F1_corr_ndvi - F2_corr_ndvi) - (F1_ndvi - F2_ndvi))
#     plt.bar(class_set, y_F1_F2_diff_F1_corr_F2_corr_ndvi)
#     plt.title('Уменьшение разницы NDVI между снимками 04.06.2019 и 06.06.2019 после коррекции sen2corr')
#     plt.xlabel('Wavelength')
#     plt.grid()
#     plt.show()
#
#     mask_map = Processing.Map(mask2_samples, features=mask_samples_features,
#                               non_classified=False, non_classified_color='darkviolet', uniform_samples='Oversampling',
#                               normalize=False)
#     mask_map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', cv=10, n_neighbors=40)
#     # mask_map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10)
#     # mask_map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#     #                      n_estimators=[100], max_depth=[20], class_weight=['balanced'])
#     # mask_map.to_fit_by_SVM(greed_search=False, search_scoring='accuracy', cv=10)
#
#     mask = mask_map.to_process_mask(F1_DOS_shot, 'Forest', spec_features=spectral_list, texture_features=texture_list,
#                                     texture_adjacency_directions_dict=text_directions_dict)
#
#     mask_map.to_save_map('mask_knn_win_80_grad_50', result_directory)
#     F1_DOS_shot.remove_by_mask(mask)
#     F2_DOS_shot.remove_by_mask(mask)
#
#     F_map = Processing.Map(F2_samples, features=samples_features,
#                          non_classified=False, non_classified_color='darkviolet', uniform_samples='Oversampling',
#                          normalize=False)
#
#     F_map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', n_neighbors=40)
#     # F_map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10)
#     # F_map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#     #                 n_estimators=[100], max_depth=[20], class_weight=['balanced'])
#     # F_map.to_fit_by_SVM(greed_search=False, search_scoring='accuracy', cv=10)
#
#     F_map.self_test('F1_knn_test', result_directory, cross_validation=True, reclassification=True, sample_remainder=False)
#
#     F_map.to_process(F1_DOS_shot, spec_features=samples_features, texture_features=[])
#
#     F_map.to_save_map('knn_map', result_directory)
#
#
#     i = 3
#     F_map.map[np.where((F_map.map[:, :, 0] == F_map.color_map[i][0]) &
#                      (F_map.map[:, :, 1] == F_map.color_map[i][1]) &
#                      (F_map.map[:, :, 2] == F_map.color_map[i][2]) &
#                      (F_map.map[:, :, 3] == F_map.color_map[i][3]))] = F_map.color_map[2]
#
#     dominant_shp_address = result_directory + '/knn_dominant.shp'
#     #dominant2_shp_address = result_directory + '/knn_dominant2.shp'
#
#     F_map.to_calc_dominant_2(test_shape_address, dominant_shp_address, 'DOM',
#                            fields_to_copy=None,
#                            space_field_name='WEIGTH', min_space=50,
#                            )
#     # map.to_calc_dominant_2(test_shape_address, dominant2_shp_address, 'DOM',
#     #                      fields_to_copy=None,
#     #                      space_field_name='WEIGTH',
#     #                      dom_order=2, min_space=10
#     #                      )
#
#     F_map.to_compare_areas(dominant_shp_address, 'SPEC', dominant_shp_address, 'DOM', 'id', 'knn_dominant_test',
#                            result_directory)
#
#     right_class_percent_shp_address = result_directory + '/correct_class_percent_knn.shp'
#
#     full_map = mask_map.add_map(F_map)
#     full_map.to_save_map('knn_full_map', result_directory)

# Проект Сербия (атмосферная коррекция)
import numpy as np

import DataShot
import DataSamples
import Processing

import shapefile
import os

from matplotlib import pyplot as plt

if __name__ == "__main__":
    spectral_list = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    model = -2
    samples_features = [
        'blue', 'green', 'red', 'nir', 'swir1', 'swir2'
    ]

    F1_shot_address = 'D:/Проекты/Велики Столак/inter_data/Shot_Sentinel_win_80_grad_50_full_aerosol_(2021.05.04).file'
    F2_shot_address = 'D:/Проекты/Велики Столак/inter_data/qgis_DOS_Shot_Sentinel_win_80_grad_50_full_aerosol_(2021.05.04).file'
    F1_train_samples_address = 'D:/Проекты/Велики Столак/inter_data/selection.file'
    F2_train_samples_address = 'D:/Проекты/Велики Столак/inter_data/qgis_DOS_selection.file'
    intermediate_data_directory = 'D:/Проекты/Велики Столак/inter_data'
    result_directory = 'D:/Проекты/Велики Столак/Результаты'

    F1_samples = DataSamples.to_load_samples_set(F1_train_samples_address)
    F2_samples = DataSamples.to_load_samples_set(F2_train_samples_address)
    F1_shot = DataShot.to_load_data_shot(F1_shot_address)
    F2_shot = DataShot.to_load_data_shot(F2_shot_address)
    F1_DOS_shot = DataShot.to_load_data_shot(F1_shot_address)
    #F2_DOS_shot = DataShot.to_load_data_shot(F2_shot_address)
    #F1_DOS_shot.DOS_colebration('aerosol', model, start_point='min', show_correction_curve=False, show_start_point=False, map_band='red', neg_rule='calib')
    F1_DOS_shot.simple_DOS_colebration()
    #F2_DOS_shot.DOS_colebration('aerosol', model, start_point='hist', show_correction_curve=False, show_start_point=False, map_band='red', neg_rule='calib')
    F1_DOS_shot.name = 'DOS_' + F1_DOS_shot.name
    #F2_DOS_shot.name = 'corr_' + F2_DOS_shot.name

    F_shape_address = 'D:/Проекты/Велики Столак/Vector/Border_2.shp'
    F1_shot.graph_compare(F1_DOS_shot, F_shape_address, spec_bands_keys=samples_features, class_field='CLASS',
                         reg_color='red', class_mean=False)
    # F2_shot.graph_compare(F2_shot, F_shape_address, spec_bands_keys=samples_features, class_field='CLASS',
    #                      reg_color='blue', class_mean=False, legend=False)
    # F2_shot.graph_compare(F1_DOS_shot, F_shape_address, spec_bands_keys=samples_features, class_field='CLASS',
    #                      reg_color='green', class_mean=False, legend=False)
    print()

# # Расчеты в задаче определения полноты части Валуйского лесничества
# import numpy as np
#
# import DataShot
# import DataSamples
# import Processing
#
# import shapefile
# import os
#
# if __name__ == "__main__":
#     spectral_list = []
#     texture_list = ['SDGL', 'Contrast', 'Entropy']
#     text_directions = [0]
#     win = 64
#     dist = [1]
#     samples_features = [
#                        'SDGL-0.0 dist_1 layer_0',
#                        'Contrast-0.0 dist_1 layer_0',
#                        'Entropy-0.0 dist_1 layer_0']
#
#     shot_address = 'D:/Проекты/Структурные индексы/Промежуточные данные/p_WorldView_shot_25.file'
#     train_samples_address = 'D:/Проекты/Структурные индексы/Промежуточные данные/p_train_selection_25.file'
#     test_samples_address = 'D:/Проекты/Структурные индексы/Промежуточные данные/p_test_selection_25.file'
#     result_directory = 'D:/Проекты/Структурные индексы/Результаты'
#
#     samples = DataSamples.to_load_samples_set(train_samples_address)
#     shot = DataShot.to_load_data_shot(shot_address)
#
#     map = Processing.Map(samples, features=samples_features,
#                         non_classified=False, non_classified_color='darkviolet', #uniform_samples='Oversampling',
#                         normalize=True)
#     map.to_fit_by_GPR(greed_search=False, search_scoring='accuracy')
#     # map.to_fit_by_LR(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#     #                  penalty=['l1'], solver=['liblinear'])
#
#     text_directions_dict = {0:{
#                            'SDGL': {1: [0]},
#                            'Contrast': {1: [0]},
#                            'Entropy': {1: [0]}
#                            }}
#
#     texture_adj_dir_dict = text_directions_dict
#     #for texture in texture_list:
#     #    texture_adj_dir_dict[texture] = {}
#     #for texture in texture_list:
#     #    for dir in dist:
#     #        texture_adj_dir_dict[texture].update({dir: text_directions})
#
#     #map.to_draw_classes('SDGL-90.0 dist_1 layer_3', 'Entropy-90.0 dist_1 layer_1',
#     #                    hyperspaces_borders=(100, 0, 0, 100), plane_step=0.01)
#
#     map.to_process(shot, spec_features=spectral_list, texture_features=texture_list,
#                   texture_adjacency_directions_dict=text_directions_dict)
#     map.to_save_map('p_fullness', result_directory)
#
#     #print(map.to_test_mashine('test', result_directory, test_sample_set=test_samples))
#
#     test_samples_address = 'D:/Проекты/Структурные индексы/Векторные данные/test_samples_reg.shp'
#     mean_shp_address = result_directory + '/p_mean_50.shp'
#     # map.to_calc_dominant(test_samples_address, dominant_shp_address, 'DOM', shot,
#     #                             fields_to_copy=None,
#     #                             #possible_dom_classes=['C', 'D'],
#     #                             texture_features=texture_list,
#     #                             texture_adjacency_directions_dict=texture_adj_dir_dict,
#     #                             )
#     map.to_calc_mean(test_samples_address, mean_shp_address, 'MEAN', shot,
#                      fields_to_copy=None,
#                      texture_features=texture_list,
#                      texture_adjacency_directions_dict=texture_adj_dir_dict,
#                      borders=[0, 1], space_field_name='WEIGTH'
#                      )
#     #map.to_compare_areas(mean_shp_address, 'CLASS', mean_shp_address, 'DOM', 'id', 'dominant_test', result_directory)
#     print(map.to_compare_areas_err(mean_shp_address, 'SKAL1', mean_shp_address, 'MEAN', 'id', weights_col_name='WEIGTH'))

# # Расчеты в задаче определения полноты части Валуйского лесничества
# import numpy as np
#
# import DataShot
# import DataSamples
# import Processing
#
# import shapefile
# import os
#
# if __name__ == "__main__":
#     spectral_list = []
#     texture_list = ['SDGL', 'Contrast', 'Entropy']
#     text_directions = [0]
#     win = 16
#
#     dist = [1]
#     samples_features = [
#                         'SDGL-0.0 dist_1 layer_0',
#                         'Contrast-0.0 dist_1 layer_0',
#                         'Entropy-0.0 dist_1 layer_0',
#                         'SDGL-0.0 dist_1 layer_1',
#                         'Contrast-0.0 dist_1 layer_1',
#                         'Entropy-0.0 dist_1 layer_1',
#                         'SDGL-0.0 dist_1 layer_2',
#                         'Contrast-0.0 dist_1 layer_2',
#                         'Entropy-0.0 dist_1 layer_2',
#                         'SDGL-0.0 dist_1 layer_3',
#                         'Contrast-0.0 dist_1 layer_3',
#                         'Entropy-0.0 dist_1 layer_3',
#                         'SDGL-0.0 dist_1 layer_4',
#                         'Contrast-0.0 dist_1 layer_4',
#                         'Entropy-0.0 dist_1 layer_4',
#                         'SDGL-0.0 dist_1 layer_5',
#                         # 'Contrast-0.0 dist_1 layer_5',
#                         'Entropy-0.0 dist_1 layer_5',
#                         'SDGL-0.0 dist_1 layer_6',
#                         'Contrast-0.0 dist_1 layer_6',
#                         'Entropy-0.0 dist_1 layer_6',
#                         'SDGL-0.0 dist_1 layer_7',
#                         # 'Contrast-0.0 dist_1 layer_7',
#                         'Entropy-0.0 dist_1 layer_7'
#                         ]
#
#     shot_address = 'D:/Проекты/Структурные индексы/Промежуточные данные/WorldView_shot_50.file'
#     train_samples_address = 'D:/Проекты/Структурные индексы/Промежуточные данные/train_selection.file'
#     test_samples_address = 'D:/Проекты/Структурные индексы/Промежуточные данные/test_selection.file'
#     result_directory = 'D:/Проекты/Структурные индексы/Результаты'
#
#     train_samples = DataSamples.to_load_samples_set(train_samples_address)
#     test_samples = DataSamples.to_load_samples_set(test_samples_address)
#     shot = DataShot.to_load_data_shot(shot_address)
#
#     map = Processing.Map(train_samples, features=samples_features,
#                          non_classified=False, non_classified_color='darkviolet', #uniform_samples='Oversampling',
#                          normalize=True)
#     map.to_fit_by_GPR(greed_search=False, search_scoring='accuracy')
#     #map.to_fit_by_lasso(greed_search=False, search_scoring='accuracy', alpha=[0.01])
#     # print(map.mashine.coef_)
#     #map.to_fit_by_LR_reg(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10)
#
#     text_directions_dict = {
#                             0:{
#                             'SDGL': {1: [0]},
#                             'Contrast': {1: [0]},
#                             'Entropy': {1: [0]}
#                             },
#         1: {
#             'SDGL': {1: [0]},
#             'Contrast': {1: [0]},
#             'Entropy': {1: [0]}
#         },
#         2: {
#             'SDGL': {1: [0]},
#             'Contrast': {1: [0]},
#             'Entropy': {1: [0]}
#         },
#         3: {
#             'SDGL': {1: [0]},
#             'Contrast': {1: [0]},
#             'Entropy': {1: [0]}
#         },
#         4: {
#              'SDGL': {1: [0]},
#              'Contrast': {1: [0]},
#              'Entropy': {1: [0]}
#         },
#         5: {
#             'SDGL': {1: [0]},
#             # 'Contrast': {1: [0]},
#             'Entropy': {1: [0]}
#         },
#         6: {
#             'SDGL': {1: [0]},
#             'Contrast': {1: [0]},
#             'Entropy': {1: [0]}
#         },
#         7: {
#             'SDGL': {1: [0]},
#             # 'Contrast': {1: [0]},
#             'Entropy': {1: [0]}
#         }
#         }
#
#     texture_adj_dir_dict = text_directions_dict
#     #for texture in texture_list:
#     #    texture_adj_dir_dict[texture] = {}
#     #for texture in texture_list:
#     #    for dir in dist:
#     #        texture_adj_dir_dict[texture].update({dir: text_directions})
#
#     map.to_process(shot, spec_features=spectral_list, texture_features=texture_list,
#                    texture_adjacency_directions_dict=text_directions_dict, borders=[0, 1])
#     map.to_save_map('fullness', result_directory)
#     #print(map.to_test_mashine('test', result_directory, test_sample_set=test_samples))
#
#     test_samples_address = 'D:/Проекты/Структурные индексы/Векторные данные/test_samples_reg.shp'
#     mean_shp_address = result_directory + '/mean.shp'
#     # map.to_calc_dominant(test_samples_address, dominant_shp_address, 'DOM', shot,
#     #                             fields_to_copy=None,
#     #                             #possible_dom_classes=['C', 'D'],
#     #                             texture_features=texture_list,
#     #                             texture_adjacency_directions_dict=texture_adj_dir_dict,
#     #                             )
#     map.to_calc_mean(test_samples_address, mean_shp_address, 'MEAN', shot,
#                                 fields_to_copy=None,
#                                 texture_features=texture_list,
#                                 texture_adjacency_directions_dict=texture_adj_dir_dict,
#                                 borders=[0, 1], space_field_name='WEIGTH'
#                                 )
#     #map.to_compare_areas(mean_shp_address, 'CLASS', mean_shp_address, 'DOM', 'id', 'dominant_test', result_directory)
#     print(map.to_compare_areas_err(mean_shp_address, 'SKAL1', mean_shp_address, 'MEAN', 'id', weights_col_name='WEIGTH'))

#import numpy as np
#
#import DataShot
#import DataSamples
#import Processing
#
#import shapefile
#import os
#
#if __name__ == "__main__":
#    spectral_list = ['blue', 'swir1', 'swir2']
#    texture_list = []
#    texture_list = [
#                    #'Autocorrelation',
#                    'ClusterShade',
#                    'Contrast',
#                   #'Correlation'
#                   ]
#    text_directions = [0, 3 * np.pi / 4]
#    win = 80
#    dist = [1]
#    grad = [100]
#    #samples_features = ['blue', 'swir1', 'swir2']
#    samples_features = ['blue', 'swir1', 'swir2',
#                        #'Autocorrelation-0.0 dist_1',
#                        'ClusterShade-0.0 dist_1',
#                        'Contrast-0.0 dist_1',
#                        #'Correlation-135.0 dist_1', 'Correlation-135.0 dist_1'
#                        ]
#    # samples_features = [51, 42, 25, 41, 48, 24, 40, 39, 47, 36, 26, 43, 50, 46, 35, 38, 27]
#
#    shot_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                        'Общие данные/Data shots/west_win_80_dist_[1]_grad_100.file'
#    shot_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                        'Общие данные/Data shots/east_win_80_dist_[1]_grad_100.file'
#    samples_address = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                       'Классификации/Классификация леса на лиственный и хвойный/Промежуточные результаты/samples.file'
#    west_test_shp_address = 'D:/Data/Лесотаксация/Forest (west).shp'
#    east_test_shp_address = 'D:/Data/Лесотаксация/Forest (east).shp'
#    result_directory = 'D:/Documents/Статьи/ЛЕ2020/Результаты'
#    west_dominant_shp_address = result_directory + '/west_dominant.shp'
#    east_dominant_shp_address = result_directory + '/east_dominant.shp'
#    west_sub_dominant_shp_address = result_directory + '/west_sub_dominant.shp'
#    east_sub_dominant_shp_address = result_directory + '/east_sub_dominant.shp'
#    # shot_east = DataShot.to_load_data_shot('D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (WorldView 2)/Общие данные/Data shots/hyper.file')
#
#    samples = DataSamples.to_load_samples_set(samples_address)
#    shot_west = DataShot.to_load_data_shot(shot_address_west)
#    shot_east = DataShot.to_load_data_shot(shot_address_east)
#
#    rf_map = Processing.Map(samples, features=samples_features,
#                            non_classified=False, non_classified_color='darkviolet', uniform_samples=False,
#                            normalize=True)
#
#    # rf_map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#    #                     n_estimators=[50], max_depth=[15], class_weight=['balanced'])
#    rf_map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#                         n_neighbors=[75])
#    #rf_map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10)
#    #text_directions_dict = {}
#    text_directions_dict = {
#                            #'Autocorrelation': {1: [0]},
#                            'ClusterShade': {1: [0]},
#                            'Contrast': {1: [0]},
#    #                        #'Correlation': {1: [0, 3 * np.pi / 4]}
#                            }
#    texture_adj_dir_dict = text_directions_dict
#    # for texture in texture_list:
#    #    texture_adj_dir_dict[texture] = {}
#    # for texture in texture_list:
#    #    for dir in dist:
#    #        texture_adj_dir_dict[texture].update({dir: text_directions})
#    rf_map.to_test_mashine('test_win_' + str(win) + '_dist_' + str(dist) + '_grad_' + str(grad), result_directory)
#
#
#    # создание карты западной части
#    rf_map.to_process(shot_west, spec_features=spectral_list, texture_features=texture_list,
#                      texture_adjacency_directions_dict=text_directions_dict)
#    rf_map.to_save_map('west_map_win_' + str(win) + '_dist_' + str(dist) + '_grad_' + str(grad), result_directory)
#    rf_map.to_process(shot_east, spec_features=spectral_list, texture_features=texture_list,
#                      texture_adjacency_directions_dict=text_directions_dict)
#    rf_map.to_save_map('east_map_win_' + str(win) + '_dist_' + str(dist) + '_grad_' + str(grad), result_directory)
#
#    rf_map.to_calc_dominant(west_test_shp_address, west_dominant_shp_address, 'M_DOM', shot_west,
#                            fields_to_copy=['TYPE_DOM', 'TYPE_SUB', 'id'],
#                            possible_dom_classes=['C', 'D'],
#                            spec_features=spectral_list,
#                            texture_features=texture_list,
#                            texture_adjacency_directions_dict=texture_adj_dir_dict,
#                            dom_order=1,
#                            min_space=60,
#                            note_field='NOTE')
#    rf_map.to_calc_dominant(east_test_shp_address, west_dominant_shp_address, 'M_DOM', shot_east,
#                            fields_to_copy=['TYPE_DOM', 'TYPE_SUB', 'id'],
#                            possible_dom_classes=['C', 'D'],
#                            spec_features=spectral_list,
#                            texture_features=texture_list,
#                            texture_adjacency_directions_dict=texture_adj_dir_dict,
#                            dom_order=1, min_space=60, note_field='NOTE')
#    rf_map.to_compare_areas(west_dominant_shp_address, 'TYPE_DOM',
#                           west_dominant_shp_address, 'M_DOM', 'id', 'west_dominant_test', result_directory)
#
#    rf_map.to_calc_dominant(west_test_shp_address, west_sub_dominant_shp_address, 'M_SUB_DOM', shot_west,
#                            fields_to_copy=['TYPE_DOM', 'TYPE_SUB', 'id'],
#                            possible_dom_classes=['C', 'D'],
#                            spec_features=spectral_list,
#                            texture_features=texture_list,
#                            texture_adjacency_directions_dict=texture_adj_dir_dict,
#                            dom_order=2)
#    rf_map.to_calc_dominant(east_test_shp_address, west_sub_dominant_shp_address, 'M_SUB_DOM', shot_east,
#                            fields_to_copy=['TYPE_DOM', 'TYPE_SUB', 'id'],
#                            possible_dom_classes=['C', 'D'],
#                            spec_features=spectral_list,
#                            texture_features=texture_list,
#                            texture_adjacency_directions_dict=texture_adj_dir_dict,
#                            dom_order=2)
#    rf_map.to_compare_areas(west_sub_dominant_shp_address, 'TYPE_SUB',
#                            west_sub_dominant_shp_address, 'M_SUB_DOM', 'id', 'west_sub_dominant_test', result_directory)
#
#    fields_to_copy = ['id', 'TYPE_DOM', 'TYPE_SUB', 'M_DOM', 'M_SUB_DOM']
#    # загрузка shape-файла с диска для чтения и редактирования
#    initial_areas = shapefile.Reader(west_sub_dominant_shp_address)
#    # создание нового shape-файла, если его нет
#    if os.path.exists(west_sub_dominant_shp_address):
#        areas_with_dominant = shapefile.Editor(west_dominant_shp_address)
#    else:
#        areas_with_dominant = shapefile.Writer(shapefile.POLYGON)
#        # копирование содержимого в новый файл
#    # areas_with_dominant._shapes.extend(initial_areas.shapes())
#    fields_to_dominants_areas = areas_with_dominant.fields + [initial_areas.fields[-1]]
#    areas_with_dominant.fields = list(fields_to_dominants_areas)
#    areas_with_dominant.records = list(np.concatenate(
#        (np.array(areas_with_dominant.records), np.array([np.array(initial_areas.records())[:, -1]]).swapaxes(0, 1)),
#        axis=1))
#    areas_with_dominant.save(west_dominant_shp_address)
#
# import numpy as np
#
# import DataShot
# import DataSamples
# import Processing
#
# import shapefile
# import os
#
# if __name__ == "__main__":
#     spectral_list = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
#     # texture_list = []
#     texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#     text_directions = [0, 3 * np.pi / 4]
#     win = 80
#     dist = [1]
#     grad = [100]
#     # samples_features = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
#     samples_features = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2',
#                         'Autocorrelation-0.0 dist_1',
#                         'ClusterShade-0.0 dist_1', 'ClusterShade-135.0 dist_1',
#                         'Contrast-0.0 dist_1', 'Contrast-135.0 dist_1',
#                         'Correlation-0.0 dist_1', 'Correlation-135.0 dist_1']
#
#     shot_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                         'Общие данные/Data shots/west_win_80_dist_[1]_grad_100.file'
#     shot_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                         'Общие данные/Data shots/east_win_80_dist_[1]_grad_100.file'
#     samples_address = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                                   'Классификации/Классификация леса на лиственный и хвойный/Промежуточные результаты/samples.file'
#     west_test_shp_address = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                             'Классификации/Классификация леса на лиственный и хвойный/Shapes/Forest (west).shp'
#     east_test_shp_address = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                             'Классификации/Классификация леса на лиственный и хвойный/Shapes/Forest (east).shp'
#     result_directory = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                        'Классификации/Классификация леса на лиственный и хвойный/Результаты'
#     west_dominant_shp_address = result_directory + '/west_dominant.shp'
#     east_dominant_shp_address = result_directory + '/east_dominant.shp'
#     # shot_east = DataShot.to_load_data_shot('D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (WorldView 2)/Общие данные/Data shots/hyper.file')
#
#     samples = DataSamples.to_load_samples_set(samples_address)
#     shot_west = DataShot.to_load_data_shot(shot_address_west)
#     shot_east = DataShot.to_load_data_shot(shot_address_east)
#
#     rf_map = Processing.Map(samples, features=samples_features,
#                             non_classified=False, non_classified_color='darkviolet', uniform_samples=False,
#                             normalize=True)
#
#     # rf_map.to_fit_by_RF(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#     #                     n_estimators=[50], max_depth=[15], class_weight=['balanced'])
#     # rf_map.to_fit_by_KNN(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10,
#     #                      n_neighbors=[75])
#     rf_map.to_fit_by_QDA(greed_search=False, search_scoring='accuracy', cv=10, k_fold_order=10)
#     # text_directions_dict = {}
#     # texture_adj_dir_dict = {}
#     text_directions_dict = {'Autocorrelation': {1: [3 * np.pi / 4]},
#                            'ClusterShade': {1: [0, 3 * np.pi / 4]},
#                            'Contrast': {1: [0, 3 * np.pi / 4]},
#                            'Correlation': {1: [0, 3 * np.pi / 4]}}
#     texture_adj_dir_dict = {'Autocorrelation': {1: [3 * np.pi / 4]},
#                            'ClusterShade': {1: [0, 3 * np.pi / 4]},
#                            'Contrast': {1: [0, 3 * np.pi / 4]},
#                            'Correlation': {1: [0, 3 * np.pi / 4]}}
#     # for texture in texture_list:
#     #    texture_adj_dir_dict[texture] = {}
#     # for texture in texture_list:
#     #    for dir in dist:
#     #        texture_adj_dir_dict[texture].update({dir: text_directions})
#     rf_map.to_test_mashine('test_win_' + str(win) + '_dist_' + str(dist) + '_grad_' + str(grad), result_directory)
#
#     # texture_adj_dir_dict = {}
#     texture_adj_dir_dict = {'Autocorrelation': {1: [3 * np.pi / 4]},
#                            'ClusterShade': {1: [0, 3 * np.pi / 4]},
#                            'Contrast': {1: [0, 3 * np.pi / 4]},
#                            'Correlation': {1: [0, 3 * np.pi / 4]}}
#
#     # создание карты западной части
#     rf_map.to_process(shot_west, spec_features=spectral_list, texture_features=texture_list,
#                       texture_adjacency_directions_dict=text_directions_dict)
#     rf_map.to_save_map('west_map_win_' + str(win) + '_dist_' + str(dist) + '_grad_' + str(grad), result_directory)
#     rf_map.to_process(shot_east, spec_features=spectral_list, texture_features=texture_list,
#                       texture_adjacency_directions_dict=text_directions_dict)
#     rf_map.to_save_map('east_map_win_' + str(win) + '_dist_' + str(dist) + '_grad_' + str(grad), result_directory)
#
#     rf_map.to_calc_dominant(west_test_shp_address, west_dominant_shp_address, 'DOM', shot_west,
#                             fields_to_copy=['Q', 'VOZ', 'DEC', 'CON', 'TYPE', 'DOM_FOREST', 'id'],
#                             # possible_dom_classes=['Coniferous', 'Deciduous'],
#                             spec_features=spectral_list,
#                             texture_features=texture_list,
#                             texture_adjacency_directions_dict=texture_adj_dir_dict,
#                             dom_order=1)
#     rf_map.to_calc_dominant(east_test_shp_address, west_dominant_shp_address, 'DOM', shot_east,
#                             fields_to_copy=['Q', 'VOZ', 'DEC', 'CON', 'TYPE', 'DOM_FOREST', 'id'],
#                             # possible_dom_classes=['Coniferous', 'Deciduous'],
#                             spec_features=spectral_list,
#                             texture_features=texture_list,
#                             texture_adjacency_directions_dict=texture_adj_dir_dict,
#                             dom_order=1)
#     rf_map.to_compare_areas(west_dominant_shp_address, 'DOM_FOREST',
#                            west_dominant_shp_address, 'DOM', 'id', 'west_dominant_test', result_directory)
