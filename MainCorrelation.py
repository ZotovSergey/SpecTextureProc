import numpy as np
import Correlation
import DataShot

from scipy.stats import pearsonr


if __name__ == "__main__":
    west_polygons_address = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
                            'Классификации/Классификация леса на лиственный и хвойный/Shapes/test_selection (west).shp'
    # east_polygons_address = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
    #                         'Классификации/Классификация леса на лиственный и хвойный/Shapes/test_selection (east).shp'
    west_shot = DataShot.to_load_data_shot('D:/Проекты/Классификация (спектальные и текстурные данные)/'
                                           'Саватьевское лес-во2/Общие данные/Data shots/Old/west_win_20_dist_1_grad_100.file')
    east_shot = DataShot.to_load_data_shot('D:/Проекты/Классификация (спектальные и текстурные данные)/'
                                           'Саватьевское лес-во2/Общие данные/Data shots/Old/east_win_20_dist_1_grad_100.file')
    intermediate_data_directory = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
                                  'Классификации/Классификация леса на лиственный и хвойный/Промежуточные результаты'

    west_high_res_im_address = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
                               'Общие данные/Rastrs/Текстурные данные/Запад/Градуированные данные/high res 100 (west).tif'
    east_high_res_im_address = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
                                'Общие данные/Rastrs/Текстурные данные/Восток/Градуированные данные/high res 100 (east).tif'
    file_name = 'between_correlation_test_2'
    balanced_file_name = 'balanced_correlation_test'

    data = Correlation.DataSet()
    # data.to_collect_data_from_image(west_high_res_im_address, west_polygons_address, distance=[1, 2, 3], mark_dist=True)
    # data.to_collect_data_from_image(east_high_res_im_address, east_polygons_address, distance=[1, 2, 3], mark_dist=True)
    # data.to_save_data_shot(file_name, intermediate_data_directory)

    x_features = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2',
                  #'Autocorrelation 0', 'Autocorrelation 45', 'Autocorrelation 90', 'Autocorrelation 135',
                  #'ClusterProminence 0', 'ClusterProminence 45', 'ClusterProminence 90', 'ClusterProminence 135',
                  #'ClusterShade 0', 'ClusterShade 45', 'ClusterShade 90', 'ClusterShade 135',
                  'Contrast 0', 'Contrast 45', 'Contrast 90', 'Contrast 135',
                  'Correlation 0', 'Correlation 45', 'Correlation 90', 'Correlation 135',
                  'DiffEntropy 0', 'DiffEntropy 45', 'DiffEntropy 90', 'DiffEntropy 135',
                  'DiffVariance 0', 'DiffVariance 45', 'DiffVariance 90', 'DiffVariance 135',
                  'Dissimilarity 0', 'Dissimilarity 45', 'Dissimilarity 90', 'Dissimilarity 135',
                  #'Energy 0', 'Energy 45', 'Energy 90', 'Energy 135',
                  #'Entropy 0', 'Entropy 45', 'Entropy 90', 'Entropy 135',
                  'Homogeneity 0', 'Homogeneity 45', 'Homogeneity 90', 'Homogeneity 135',
                  'Homogeneity2 0', 'Homogeneity2 45', 'Homogeneity2 90', 'Homogeneity2 135',
                  'InfMeasureCorr1 0', 'InfMeasureCorr1 45', 'InfMeasureCorr1 90', 'InfMeasureCorr1 135',
                  'InfMeasureCorr2 0', 'InfMeasureCorr2 45', 'InfMeasureCorr2 90', 'InfMeasureCorr2 135',
                  #'MaxProb 0', 'MaxProb 45', 'MaxProb 90', 'MaxProb 135',
                  #'SumAverage 0', 'SumAverage 45', 'SumAverage 90', 'SumAverage 135',
                  #'SumEntropy 0', 'SumEntropy 45', 'SumEntropy 90', 'SumEntropy 135',
                  #'SumSquares 0', 'SumSquares 45', 'SumSquares 90', 'SumSquares 135',
                  #'SumVariance 0', 'SumVariance 45', 'SumVariance 90', 'SumVariance 135'
                  ]

    y_feature = 'TYPE'

    data = DataShot.to_load_data_shot(intermediate_data_directory + '/' + file_name + '.file')
    data.to_balance_classes('CLASS')
    # data.to_save_data_shot(balanced_file_name, intermediate_data_directory)

    print(data.correlation_among_themselves(['Autocorrelation 0 dist_1',
                                             'SumAverage 0 dist_1',
                                             'ClusterProminence 0 dist_1',
                                             'Correlation 0 dist_1',
                                             'Energy 0 dist_1',
                                             'Entropy 0 dist_1',
                                             'InfMeasureCorr1 0 dist_1',
                                             'InfMeasureCorr2 0 dist_1',
                                             'MaxProb 0 dist_1',
                                             'SumEntropy 0 dist_1',
                                             'SumSquares 0 dist_1',
                                             'SumVariance 0 dist_1',
                                             'ClusterShade 0 dist_1',
                                             'Contrast 0 dist_1',
                                             'DiffEntropy 0 dist_1',
                                             'DiffVariance 0 dist_1',
                                             'Dissimilarity 0 dist_1',
                                             'Homogeneity2 0 dist_1',
                                             'Homogeneity 0 dist_1'
                                             ],
                                              method='spearman',
                                              plot_graph=True,
                                              ticks_names = #None))
                                                         ['Автокорреляция',
                                                          'Среднее суммы',
                                                          'Островершинность',
                                                          'Корреляция',
                                                          'Энергия',
                                                          'Энтропия',
                                                          'Инфо-ая мера корр-ции 1',
                                                          'Инфо-ая мера корр-ции 2',
                                                          'Максимальная вероятность',
                                                          'Энтропия суммы',
                                                          'Сумма квадратов',
                                                          'Дисперсия суммы',
                                                          'Асимметрия',
                                                          'Контраст',
                                                          'Энтропия разности',
                                                          'Дисперсия разности',
                                                          'Неоднородность',
                                                          'Однородность',
                                                          'Однородность2'
                                                          ]))
                                                            #['Autocorrelation',
                                                            # 'Sum average',
                                                            # 'Cluster prominence',
                                                            # 'Correlation',
                                                            # 'Energy',
                                                            # 'Entropy',
                                                            # 'Inf. measure of corr. 1',
                                                            # 'Inf. measure of corr. 2',
                                                            # 'Maximum probability',
                                                            # 'Sum entropy',
                                                            # 'Sum of squares',
                                                            # 'Sum variance',
                                                            # 'Cluster shade',
                                                            # 'Contrast',
                                                            # 'Difference entropy',
                                                            ## 'Difference variance',
                                                            # 'Dissimilarity',
                                                            # 'Homogeneity',
                                                            # 'Inverse difference'
                                                            # ]))

    text_num = 'ClusterShade'
    text = 'Correlation'
    print(text_num)

    data.correlation_among_themselves([text_num + ' 0 dist_1',
                                     text_num + ' 45 dist_1',
                                     text_num + ' 90 dist_1',
                                     text_num + ' 135 dist_1'], method='spearman', plot_graph=True,
                                      ticks_names=
                                      ['0°',
                                       '45°',
                                       '90°',
                                       '135°'])

    data.correlation_among_themselves([text_num + ' 0 dist_1',
                                       text_num + ' 0 dist_2',
                                       text_num + ' 0 dist_3'], method='spearman', plot_graph=True,
                                      ticks_names=
                                      ['1',
                                       '2',
                                       '3']
                                      )

    # # print('     \t' + '0' + '     \t' + '45' + '     \t' + '90' + '     \t' + '135')
    # # for i, corr_str in enumerate(corr_list):
    # #     print('dist_' + str(i + 1) + '     \t' + str(round(corr_str[0], 3)) + '     \t' + str(round(corr_str[1], 3)) + '     \t' +
    # #           str(round(corr_str[2], 3)) + '     \t' + str(round(corr_str[3], 3)))
    #
    # A = data.correlation_among_themselves([text_num + ' 0 dist_1',
    #                                        text_num + ' 45 dist_1',
    #                                        text_num + ' 90 dist_1',
    #                                        text_num + ' 135 dist_1'], method='spearman').values
    # B = data.correlation_among_themselves([text_num + ' 0 dist_1',
    #                                        text_num + ' 0 dist_2',
    #                                        text_num + ' 0 dist_3'], method='spearman').values
    # print('dir')
    # for i in A:
    #     print(str(round(i[0], 3)) + '     \t' + str(round(i[1], 3)) + '     \t' + str(round(i[2], 3)) + '     \t' + str(round(i[3], 3)))
    # print('dist')
    # for i in B:
    #     print(str(round(i[0], 3)) + '     \t' + str(round(i[1], 3)) + '     \t' + str(round(i[2], 3)))
    #
    # # data1.to_calc_pearson_correlation([text_num + ' 0 dist_1'], text_num + ' 0 dist_1', print_p_value=True)
    # #
    # # print(data1.p_value_corr(text_num + ' 0 dist_1', text_num + ' 0 dist_3'))
    # #
    # # print(data1.bootstrap_confidence_intervals(text_num + ' 0 dist_1', text_num + ' 135 dist_1',
    # #                                            return_is_intersection=False))
    # # print(data1.bootstrap_confidence_intervals(text_num + ' 0 dist_1', text_num + ' 90 dist_1',
    # #                                            return_is_intersection=True))
    # data.plot_corr(text_num + ' 0 dist_1', 'Correlation' + ' 0 dist_1', class_column='CLASS', colors_dict={'Water': 'blue',
    #                                                                                                     'Grass': 'limegreen',
    #                                                                                                     'Forest': 'green',
    #                                                                                                     'Sand': 'y',
    #                                                                                                     'Town': 'darkorange'})

    # data.to_calc_mutual_info_score(x_features, y_feature)
    # data.to_make_correlation_graph('blue', y_feature)
    # data.to_make_correlation_graph('red', y_feature)
    # data.to_make_correlation_graph('green', y_feature)
    # data.to_make_correlation_graph('nir', y_feature)
    # data.to_make_correlation_graph('swir1', y_feature)
    # data.to_make_correlation_graph('swir2', y_feature)
    # data.to_make_correlation_graph('Autocorrelation 0', y_feature)
    # data.to_make_correlation_graph('ClusterProminence 0', y_feature)
    # data.to_make_correlation_graph('ClusterShade 0', y_feature)
    # data.to_make_correlation_graph('Contrast 0', y_feature)
    # data.to_make_correlation_graph('Correlation 0', y_feature)
    # data.to_make_correlation_graph('DiffEntropy 0', y_feature)
    # data.to_make_correlation_graph('DiffVariance 0', y_feature)
    # data.to_make_correlation_graph('Dissimilarity 0', y_feature)
    # data.to_make_correlation_graph('Energy 0', y_feature)
    # data.to_make_correlation_graph('Entropy 0', y_feature)
    # data.to_make_correlation_graph('Homogeneity 0', y_feature)
    # data.to_make_correlation_graph('Homogeneity2 0', y_feature)
    # data.to_make_correlation_graph('InfMeasureCorr1 0', y_feature)
    # data.to_make_correlation_graph('InfMeasureCorr2 0', y_feature)
    # data.to_make_correlation_graph('MaxProb 0', y_feature)
    # data.to_make_correlation_graph('SumAverage 0', y_feature)
    # data.to_make_correlation_graph('SumEntropy 0', y_feature)
    # data.to_make_correlation_graph('SumSquares 0', y_feature)
    # data.to_make_correlation_graph('SumVariance 0', y_feature)


    #y_feature = 'DiffVariance 45'
    #data.to_calc_pearson_correlation(['Autocorrelation 90',
    #                                  'ClusterShade 45',
    #                                  'Contrast 45',
    #                                  'DiffEntropy 45',
    #                                  'DiffVariance 45',
    #                                  'Dissimilarity 45',
    #                                  'Entropy 45',
    #                                  'Homogeneity 90',
    #                                  'Homogeneity2 45',
    #                                  'SumAverage 45',
    #                                  'SumEntropy 90'], y_feature)


    #left_polygons_address = 'D:/Проекты/Текстуры/Shape/Q_C_left.shp'
    #right_polygons_address = 'D:/Проекты/Текстуры/Shape/Q_C_right.shp'
    #left_polygons_address_add = 'D:/Проекты/Текстуры/Shape/Q_D_left.shp'
    #right_polygons_address_add = 'D:/Проекты/Текстуры/Shape/Q_D_right.shp'
    ##left_polygons_address = 'D:/Проекты/Текстуры/Shape/visual_completeness (left).shp'
    ##right_polygons_address = 'D:/Проекты/Текстуры/Shape/visual_completeness (right).shp'
    #intermediate_data_directory = 'D:/Проекты/Текстуры/Промежуточные результаты/normed'
    #shot_left = DataShot.to_load_data_shot('D:/Проекты/Текстуры/Промежуточные результаты/shot_left_100_dist_1.file')
    #shot_right = DataShot.to_load_data_shot('D:/Проекты/Текстуры/Промежуточные результаты/shot_right_100_dist_1.file')
    #data = Correlation.DataSet()
    ##data.to_collect_data_from_datashot(shot_left, left_polygons_address, row_on_polygon=True)
    ##data.to_collect_data_from_datashot(shot_right, right_polygons_address, row_on_polygon=True)
    ##data.to_save_data_shot('corr_100_many_pol_win_50_dist_1', intermediate_data_directory)

    #texture_shot_address_left = 'D:/Data/Высокое разрешение/Савватьевское лесничество/056009302010_01_P002_PAN/' \
    #                            '16JUN25085338-P2AS-056009302010_01_P002.TIF'
    #texture_shot_address_right = 'D:/Data/Высокое разрешение/Савватьевское лесничество/056009302010_01_P001_PAN/' \
    #                             '16JUN25085327-P2AS-056009302010_01_P001.TIF'
    ##texture_shot_address_left = 'D:/Проекты/Текстуры/Растр/high res 100 (left).tif'
    ##texture_shot_address_right = 'D:/Проекты/Текстуры/Растр/high res 100 (right).tif'
    #window = 20

    #data.to_collect_data_from_datashot(shot_left, left_polygons_address)
    #data.to_collect_data_from_datashot(shot_left, left_polygons_address_add)
    #data.to_collect_data_from_datashot(shot_right, left_polygons_address)
    #data.to_collect_data_from_datashot(shot_right, left_polygons_address_add)
    #name = 'forest_type_corr'
    #data.to_save_data_shot(name, intermediate_data_directory)

    ##for win in [window]:
    ##    for dist in [1]:
    ##        data = Correlation.DataSet()
    ##        data.to_collect_data_from_image(texture_shot_address_left, left_polygons_address, grad_count=[60], distance=dist,
    ##                                        window_width=win, polygons_features=None, row_on_polygon=True,
    ##                                        texture_adjacency_directions_dict=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], texture_features=['Contrast', 'DiffEntropy', 'DiffVariance', 'Dissimilarity', 'Energy', 'Entropy', 'Homogeneity', 'Homogeneity2', 'InfMeasureCorr2', 'MaxProb', 'SumEntropy'])#, texture_features=['Correlation', 'InfMeasureCorr1', 'InfMeasureCorr2'])
    ##        data.to_collect_data_from_image(texture_shot_address_right, right_polygons_address, grad_count=[60], distance=dist,
    ##                                        window_width=win, polygons_features=None, row_on_polygon=True,
    ##                                        texture_adjacency_directions_dict=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], texture_features=['Contrast', 'DiffEntropy', 'DiffVariance', 'Dissimilarity', 'Energy', 'Entropy', 'Homogeneity', 'Homogeneity2', 'InfMeasureCorr2', 'MaxProb', 'SumEntropy'])#, texture_features=['Correlation', 'InfMeasureCorr1', 'InfMeasureCorr2'])
    ##        address = 'normed_corr_100_diff_win_' + str(win) + '_dist_' + str(dist) + '_grad_test'
    ##        data.to_save_data_shot(address, intermediate_data_directory)


    #x_features = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2',
    ## x_features = [
    #              'Autocorrelation 0', 'Autocorrelation 45', 'Autocorrelation 90', 'Autocorrelation 135',
    #              'ClusterProminence 0', 'ClusterProminence 45', 'ClusterProminence 90', 'ClusterProminence 135',
    #              'ClusterShade 0', 'ClusterShade 45', 'ClusterShade 90', 'ClusterShade 135',
    #              'Contrast 45', 'Contrast 45', 'Contrast 90', 'Contrast 135',
    #              'Correlation 0', 'Correlation 45', 'Correlation 90', 'Correlation 135',
    #              'DiffEntropy 45','DiffEntropy 45', 'DiffEntropy 90', 'DiffEntropy 135',
    #              'DiffVariance 45', 'DiffVariance 45', 'DiffVariance 90', 'DiffVariance 135',
    #              'Dissimilarity 0', 'Dissimilarity 45', 'Dissimilarity 90', 'Dissimilarity 135',
    #              'Energy 45', 'Energy 45', 'Energy 90', 'Energy 135',
    #              'Entropy 45', 'Entropy 45', 'Entropy 90', 'Entropy 135',
    #              'Homogeneity 45', 'Homogeneity 45', 'Homogeneity 90', 'Homogeneity 135',
    #              'Homogeneity2 45', 'Homogeneity2 45', 'Homogeneity2 90', 'Homogeneity2 135',
    #              'InfMeasureCorr1 0', 'InfMeasureCorr1 45', 'InfMeasureCorr1 90', 'InfMeasureCorr1 135',
    #              'InfMeasureCorr2 135', 'InfMeasureCorr2 45', 'InfMeasureCorr2 90', 'InfMeasureCorr2 135',
    #              'MaxProb 45', 'MaxProb 45', 'MaxProb 90', 'MaxProb 135',
    #              'SumAverage 0', 'SumAverage 45', 'SumAverage 90', 'SumAverage 135',
    #              'SumEntropy 45', 'SumEntropy 45', 'SumEntropy 90', 'SumEntropy 135',
    #              'SumSquares 0', 'SumSquares 45', 'SumSquares 90', 'SumSquares 135',
    #              'SumVariance 0', 'SumVariance 45', 'SumVariance 90', 'SumVariance 135'
    #              ]
    #y_feature = 'red'
    ## y_feature = 'VOZ'
    ## y_feature = 'CON'

    #data = DataShot.to_load_data_shot('D:/Проекты/Текстуры/Промежуточные результаты/normed/' + name + '.file')
    ##data.to_calc_pearson_correlation(x_features, y_feature)
    #y_feature = 'TYPE'
    #data.to_calc_pearson_correlation(x_features, y_feature)
    ##data.to_make_correlation_graph('Correlation 0', y_feature)
    ##data.to_make_correlation_graph('SumEntropy 90', y_feature)
    ##data.to_make_correlation_graph('Correlation 90', y_feature)
    ##data.to_make_correlation_graph('Correlation 135', y_feature)
