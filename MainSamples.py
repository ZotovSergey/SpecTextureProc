# # Тестирование адаптивного порога
#
# from numpy import pi
#
# import DataSamples
#
# if __name__ == "__main__":
#     spectral_shot_address = 'D:/Проекты/Бронницы/Растровые данные/058041098010_01/058041098010_01_P001_MUL/' \
#                             '11JUL28090720-M2AS-058041098010_01_P001.XML'
#     texture_shot_address = 'D:/Проекты/Бронницы/Растровые данные/graded_texture_shot_50.tif'
#     shape_address = 'D:/Проекты/Бронницы/Векторные данные/test_sel.shp'
#     intermediate_data_directory = 'D:/Проекты/Бронницы/Промежуточные данные'
#     texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#     #texture_list = ['Autocorrelation']
#
#     window = 80
#     text_directions = [0]
#     dist = 1
#
#     samples = DataSamples.ClassSamplesSet(shape_address, spectral_shot_address, texture_shot_address,
#                                           'CLASS',
#                                           #['BAND_B', 'BAND_G', 'BAND_Y', 'BAND_R', 'BAND_RE', 'BAND_N', 'BAND_N2'],
#                                           #['blue', 'green', 'yellow', 'red', 'red_edge', 'nir', 'nir2'],
#                                           ['BAND_B', 'BAND_G', 'BAND_R', 'BAND_N', 'BAND_N2'],
#                                           ['blue', 'green', 'red', 'nir', 'nir2'],
#                                           texture_list=texture_list,
#                                           texture_adjacency_directions=text_directions,
#                                           window_width=window, distance=dist,
#                                           texture_data_linked_to_spec=True, average=False,
#                                           sat_name='WorldView_2',
#                                           exclude_shadow=True, offset=1
#                                           )
#
#     samples.to_recolor({
#                         'Forest': 'green',
#                         'Field': 'yellow',
#                         })
#
#     samples.to_save_samples_set('test_sel', intermediate_data_directory)

# # Проект Бронницы (WorldView)
#
# from numpy import pi
#
# import DataSamples
#
# if __name__ == "__main__":
#     spectral_shot_address = 'D:/Проекты/Бронницы/Растровые данные/058041098010_01/058041098010_01_P001_MUL/' \
#                             '11JUL28090720-M2AS-058041098010_01_P001.XML'
#     texture_shot_address = 'D:/Проекты/Бронницы/Растровые данные/graded_texture_shot_50.tif'
#     shape_address = 'D:/Проекты/Бронницы/Векторные данные/train_selection_sentinel.shp'
#     mask_shape_address = 'D:/Проекты/Бронницы/Векторные данные/mask_train_selection_WorldView.shp'
#     mask2_shape_address = 'D:/Проекты/Бронницы/Векторные данные/mask22_train_selection_WorldView.shp'
#     intermediate_data_directory = 'D:/Проекты/Бронницы/Промежуточные данные'
#     texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#
#     window = 80
#     text_directions = [0]
#     dist = 1
#
#     mask_samples = DataSamples.ClassSamplesSet(mask_shape_address, spectral_shot_address, texture_shot_address,
#                                                'CLASS',
#                                                ['BAND_B', 'BAND_G', 'BAND_Y', 'BAND_R', 'BAND_N', 'BAND_N2'],
#                                                ['blue', 'green', 'yellow', 'red', 'nir', 'nir2'],
#                                                # ['BAND_B', 'BAND_G', 'BAND_Y', 'BAND_R', 'BAND_RE', 'BAND_N', 'BAND_N2'],
#                                                # ['blue', 'green', 'yellow', 'red', 'red_edge', 'nir', 'nir2'],
#                                                texture_list=texture_list,
#                                                texture_adjacency_directions=text_directions,
#                                                window_width=window, distance=dist,
#                                                texture_data_linked_to_spec=True, average=False,
#                                                sat_name='WorldView_2')
#
#     # mask_samples.to_recolor({
#     #     'Water': 'blue',
#     #     'Field': 'lime',
#     #     'Forest': 'green',
#     #     'Build': 'gray'
#     # })
#     #
#     # mask_samples.to_save_samples_set('mask_train_selection_win_80_grad_50', intermediate_data_directory)
#
#     mask_samples.to_recolor({
#         'Water': 'blue',
#         'Field': 'lime',
#         'Forest': 'green',
#         'Build': 'gray',
#         #'Shrubs': 'yellow',
#         #'Shadow': 'midnightblue'
#     })
#
#     mask_samples.to_save_samples_set('mask_train_selection_win_80_grad_50', intermediate_data_directory)
#
#     mask2_samples = DataSamples.ClassSamplesSet(mask2_shape_address, spectral_shot_address, texture_shot_address,
#                                           'CLASS',
#                                           ['BAND_B', 'BAND_G', 'BAND_Y', 'BAND_R', 'BAND_N', 'BAND_N2'],
#                                           ['blue', 'green', 'yellow', 'red', 'nir', 'nir2'],
#                                           # ['BAND_B', 'BAND_G', 'BAND_Y', 'BAND_R', 'BAND_RE', 'BAND_N', 'BAND_N2'],
#                                           # ['blue', 'green', 'yellow', 'red', 'red_edge', 'nir', 'nir2'],
#                                           texture_adjacency_directions=text_directions,
#                                           window_width=window, distance=dist,
#                                           texture_data_linked_to_spec=True, average=False,
#                                            sat_name='WorldView_2',
#                                           exclude_shadow=False)
#
#     mask2_samples.to_recolor({
#         'Shrubs': 'y',
#         'Shadow': 'black',
#         'Forest': 'green'
#     })
#
#     mask2_samples.to_save_samples_set('mask2_train_selection_win_80_grad_50', intermediate_data_directory)
#
#     samples = DataSamples.ClassSamplesSet(shape_address, spectral_shot_address, texture_shot_address,
#                                           'CLASS',
#                                           ['BAND_B', 'BAND_G', 'BAND_Y', 'BAND_R', 'BAND_N', 'BAND_N2'],
#                                           ['blue', 'green', 'yellow', 'red', 'nir', 'nir2'],
#                                           # ['BAND_B', 'BAND_G', 'BAND_Y', 'BAND_R', 'BAND_RE', 'BAND_N', 'BAND_N2'],
#                                           # ['blue', 'green', 'yellow', 'red', 'red_edge', 'nir', 'nir2'],
#                                           texture_list=texture_list,
#                                           texture_adjacency_directions=text_directions,
#                                           window_width=window, distance=dist,
#                                           texture_data_linked_to_spec=True, average=False,
#                                           sat_name='WorldView_2',
#                                           exclude_shadow=True, offset=1)
#
#     samples.to_recolor({
#                         'O': 'green',
#                         'P': 'cyan',
#                         'S': 'dodgerblue',
#                         'A': 'red',
#                         'B': 'blueviolet',
#                         'L': 'orange',
#                         'LE': 'brown',
#                         'LIN': 'magenta'
#                         })
#
#     samples.to_save_samples_set('train_selection_win_80_grad_50', intermediate_data_directory)

# # Проект Бронницы (Resurs-P)
#
# from numpy import pi
#
# import DataSamples
#
# if __name__ == "__main__":
#     spectral_shot_addresses = 4 * ['D:/Проекты/Бронницы/Растровые данные/Ресурс-П (14.05.2018)/Resurs-P(32637).tif']
#     # spectral_shot_addresses = ['D:/Проекты/Бронницы/Растровые данные/'
#     #                            'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB_N02.04/IMG_DATA/'
#     #                            'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB_B02.jp2',
#     #                            'D:/Проекты/Бронницы/Растровые данные/'
#     #                            'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB_N02.04/IMG_DATA/'
#     #                            'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB_B03.jp2',
#     #                            'D:/Проекты/Бронницы/Растровые данные/'
#     #                            'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB_N02.04/IMG_DATA/'
#     #                            'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB_B04.jp2',
#     #                            'D:/Проекты/Бронницы/Растровые данные/'
#     #                            'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB_N02.04/IMG_DATA/'
#     #                            'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB_B08.jp2',
#     #                            'D:/Проекты/Бронницы/Растровые данные/'
#     #                            'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB_N02.04/IMG_DATA/'
#     #                            'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB_B11.jp2',
#     #                            'D:/Проекты/Бронницы/Растровые данные/'
#     #                            'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB_N02.04/IMG_DATA/'
#     #                            'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB_B12.jp2',
#     #                            ]
#     texture_shot_address = 'D:/Проекты/Бронницы/Растровые данные/graded_texture_shot_50.tif'
#     shape_address = 'D:/Проекты/Бронницы/Векторные данные/train_selection_sentinel.shp'
#     mask_shape_address = 'D:/Проекты/Бронницы/Векторные данные/mask_train_selection_resurs.shp'
#     mask2_shape_address = 'D:/Проекты/Бронницы/Векторные данные/mask2_train_selection_resurs.shp'
#     intermediate_data_directory = 'D:/Проекты/Бронницы/Промежуточные данные'
#     texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#     #texture_list = ['Autocorrelation']
#
#     window = 80
#     text_directions = [0]
#     dist = 1
#
#     mask_samples = DataSamples.ClassSamplesSet(mask_shape_address, spectral_shot_addresses, texture_shot_address,
#                                                'CLASS',
#                                                ['B1', 'B2', 'B3', 'B4'],
#                                                ['blue', 'green', 'red', 'nir'],
#                                                texture_list=texture_list,
#                                                texture_adjacency_directions=text_directions,
#                                                layer=[0, 1, 2, 3],
#                                                window_width=window, distance=dist,
#                                                texture_data_linked_to_spec=True, average=False,
#                                                sat_name='images', accurate_pol=True, text_accurate_pol=False)
#
#     mask_samples.to_recolor({
#         'Field': 'lime',
#         'Forest': 'green'
#     })
#
#     mask_samples.to_save_samples_set('mask_train_selection_win_80_grad_50', intermediate_data_directory)
#
#     mask2_samples = DataSamples.ClassSamplesSet(mask2_shape_address, spectral_shot_addresses, texture_shot_address,
#                                                'CLASS',
#                                                ['B1', 'B2', 'B3', 'B4'],
#                                                ['blue', 'green', 'red', 'nir'],
#                                                texture_list=texture_list,
#                                                texture_adjacency_directions=text_directions,
#                                                layer=[0, 1, 2, 3],
#                                                window_width=window, distance=dist,
#                                                texture_data_linked_to_spec=True, average=False,
#                                                sat_name='images', accurate_pol=True, text_accurate_pol=False)
#
#     mask2_samples.to_recolor({
#         'Shrubs': 'y',
#         'Shadow': 'darkslategray',
#         'Forest': 'green'
#     })
#
#     mask2_samples.to_save_samples_set('mask2_train_selection_win_80_grad_50', intermediate_data_directory)
#
#     samples = DataSamples.ClassSamplesSet(shape_address, spectral_shot_addresses, texture_shot_address,
#                                           'CLASS',
#                                           ['B1', 'B2', 'B3', 'B4'],
#                                           ['blue', 'green', 'red', 'nir'],
#                                           texture_list=texture_list,
#                                           texture_adjacency_directions=text_directions,
#                                           layer=[0, 1, 2, 3],
#                                           window_width=window, distance=dist,
#                                           texture_data_linked_to_spec=True, average=False,
#                                           sat_name='images',
#                                           exclude_shadow=True, offset=1, accurate_pol=True, text_accurate_pol=False)
#
#     samples.to_recolor({
#                         'O': 'green',
#                         'P': 'cyan',
#                         'S': 'dodgerblue',
#                         'A': 'red',
#                         'B': 'm',
#                         'L': 'orange'
#                         })
#
#     samples.to_save_samples_set('train_selection_win_80_grad_50', intermediate_data_directory)

# # Проект Бронницы (simp Sentinel)
#
# from numpy import pi
#
# import DataSamples
#
# if __name__ == "__main__":
#     shots_pathes = [
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.09.13)/GRANULE/L1C_T37UDB_A032522_20210913T084310/IMG_DATA/T37UDB_20210913T083601',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.07.20)/GRANULE/L1C_T37UDB_A022827_20210720T084434/IMG_DATA/T37UDB_20210720T083559',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.07.18)/GRANULE/L1C_T37UDB_A031707_20210718T085326/IMG_DATA/T37UDB_20210718T084601',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.07.13)/GRANULE/L1C_T37UDB_A022727_20210713T085013/IMG_DATA/T37UDB_20210713T084559',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.06.23)/GRANULE/L1C_T37UDB_A022441_20210623T085142/IMG_DATA/T37UDB_20210623T084559',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.06.20)/GRANULE/L1C_T37UDB_A022398_20210620T084436/IMG_DATA/T37UDB_20210620T083559',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.06.18)/GRANULE/L1C_T37UDB_A031278_20210618T085323/IMG_DATA/T37UDB_20210618T084601',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.06.03)/GRANULE/L1C_T37UDB_A022155_20210603T085145/IMG_DATA/T37UDB_20210603T084559',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2020.10.28)/GRANULE/L1C_T37UDB_A027946_20201028T084315/IMG_DATA/T37UDB_20201028T084051',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2020.10.01)/GRANULE/L1C_T37UDB_A027560_20201001T085325/IMG_DATA/T37UDB_20201001T084801',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2020.09.26)/GRANULE/L1C_T37UDB_A018580_20200926T085042/IMG_DATA/T37UDB_20200926T084719',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2020.09.25)/GRANULE/L1C_T37UDB_A018537_20200923T083655/IMG_DATA/T37UDB_20200923T083659',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2020.09.06)/GRANULE/L1C_T37UDB_A018294_20200906T084919/IMG_DATA/T37UDB_20200906T084559',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2020.06.10)/GRANULE/L1C_T37UDB_A025944_20200610T083928/IMG_DATA/T37UDB_20200610T083611',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2019.08.20)/GRANULE/L1C_T37UDB_A012817_20190820T084332/IMG_DATA/T37UDB_20190820T083609',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2019.06.06)/GRANULE/L1C_T37UDB_A020653_20190606T083602/IMG_DATA/T37UDB_20190606T083601',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2019.06.04)/GRANULE/L1C_T37UDB_A011716_20190604T085141/IMG_DATA/T37UDB_20190604T084609',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.10.17)/GRANULE/L1C_T37UDB_A008427_20181017T085347/IMG_DATA/T37UDB_20181017T084939',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.10.14)/GRANULE/L1C_T37UDB_A008384_20181014T083913/IMG_DATA/T37UDB_20181014T083919',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.10.09)/GRANULE/L1C_T37UDB_A017221_20181009T083850/IMG_DATA/T37UDB_20181009T083841',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.09.22)/GRANULE/L1C_T37UDB_A016978_20180922T085137/IMG_DATA/T37UDB_20180922T084641',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.09.02)/GRANULE/L1C_T37UDB_A016692_20180902T085218/IMG_DATA/T37UDB_20180902T084601',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.08.25)/GRANULE/L1C_T37UDB_A007669_20180825T084235/IMG_DATA/T37UDB_20180825T083549',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.08.10)/GRANULE/L1C_T37UDB_A016363_20180810T084438/IMG_DATA/T37UDB_20180810T083601',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.05.25)/GRANULE/L1C_T37UDB_A015262_20180525T085353/IMG_DATA/T37UDB_20180525T084601',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.05.12)/GRANULE/L1C_T37UDB_A015076_20180512T084250/IMG_DATA/T37UDB_20180512T083601',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.05.10)/GRANULE/L1C_T37UDB_A006139_20180510T084756/IMG_DATA/T37UDB_20180510T084559',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2017.09.24)/GRANULE/L1C_T37UDB_A011787_20170924T083955/IMG_DATA/T37UDB_20170924T083701',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2017.07.29)/GRANULE/L1C_T37UDB_A010972_20170729T085325/IMG_DATA/T37UDB_20170729T085021',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2017.05.07)/GRANULE/L1C_T37UDB_A009785_20170507T083725/IMG_DATA/T37UDB_20170507T083601',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2016.09.06)/GRANULE/S2A_OPER_MSI_L1C_TL_EPA__20160906T172349_A000676_T37UDB_N02.04/IMG_DATA/S2A_OPER_MSI_L1C_TL_EPA__20160906T172349_A000676_T37UDB',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2016.07.24)/IMG_DATA/S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2015.09.18)/GRANULE/L1C_T37UDB_A001248_20150918T085004/IMG_DATA/T37UDB_20150918T084736',
#                     # 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2015.08.09)/GRANULE/L1C_T37UDB_A000676_20150809T085008/IMG_DATA/T37UDB_20150809T085006',
#                     'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2020.11.17)/GRANULE/L1C_T37UDB_A028232_20201117T084312/IMG_DATA/T37UDB_20201117T084241',
#                     'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2019.11.23)/GRANULE/L1C_T37UDB_A023084_20191123T084306/IMG_DATA/T37UDB_20191123T084301',
#                     'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.11.23)/GRANULE/L1C_T37UDB_A008956_20181123T084251/IMG_DATA/T37UDB_20181123T084249',
#                     'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.11.16)/GRANULE/L1C_T37UDB_A008856_20181116T085413/IMG_DATA/T37UDB_20181116T085229',
#                     'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.11.06)/GRANULE/L1C_T37UDB_A008713_20181106T085139/IMG_DATA/T37UDB_20181106T085139',
#                     'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2017.11.08)/GRANULE/L1C_T37UDB_A003522_20171108T084135/IMG_DATA/T37UDB_20171108T084139'
#                     ]
#
#     prefixes = [
#                 # ' (2021.09.13)',
#                 # ' (2021.07.20)',
#                 # ' (2021.07.18)',
#                 # ' (2021.07.13)',
#                 # ' (2021.06.23)',
#                 # ' (2021.06.20)',
#                 # ' (2021.06.18)',
#                 # ' (2021.06.03)',
#                 # ' (2020.10.28)',
#                 # ' (2020.10.01)',
#                 # ' (2020.09.26)',
#                 # ' (2020.09.25)',
#                 # ' (2020.09.06)',
#                 # ' (2020.06.10)',
#                 # ' (2019.08.20)',
#                 # ' (2019.06.06)',
#                 # ' (2019.06.04)',
#                 # ' (2018.10.17)',
#                 # ' (2018.10.14)',
#                 # ' (2018.10.09)',
#                 # ' (2018.09.22)',
#                 # ' (2018.09.02)',
#                 # ' (2018.08.25)',
#                 # ' (2018.08.10)',
#                 # ' (2018.05.25)',
#                 # ' (2018.05.12)',
#                 # ' (2018.05.10)',
#                 # ' (2017.09.24)',
#                 # ' (2017.07.29)',
#                 # ' (2017.05.07)',
#                 # ' (2016.09.06)',
#                 # ' (2016.07.24)',
#                 # ' (2015.09.18)',
#                 # ' (2015.08.09)',
#                 ' (2020.11.17)',
#                 ' (2019.11.23)',
#                 ' (2018.11.23)',
#                 ' (2018.11.16)',
#                 ' (2018.11.06)',
#                 ' (2017.11.08)'
#                 ]
#
#     texture_shot_address = 'D:/Проекты/Бронницы/Растровые данные/graded_texture_shot_50.tif'
#     shape_address = 'D:/Проекты/Бронницы/Векторные данные/train_selection_sentinel_D2.shp'
#     mask_shape_address = 'D:/Проекты/Бронницы/Векторные данные/mask_train_selection_resurs.shp'
#     mask2_shape_address = 'D:/Проекты/Бронницы/Векторные данные/mask2_train_selection_sentinel.shp'
#     intermediate_data_directory = 'D:/Проекты/Бронницы/Промежуточные данные'
#     texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#
#     window = 80
#     text_directions = [0]
#     dist = 1
#
#     for i, path in enumerate(shots_pathes):
#         prefix = prefixes[i]
#
#         spectral_shot_addresses = [path + '_B02.jp2',
#                                    path + '_B03.jp2',
#                                    path + '_B04.jp2',
#                                    path + '_B08.jp2',
#                                    path + '_B11.jp2',
#                                    path + '_B12.jp2']
#
#         mask_samples = DataSamples.ClassSamplesSet(mask_shape_address, spectral_shot_addresses, texture_shot_address,
#                                                    'CLASS',
#                                                    ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                                                    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                                    texture_list=texture_list,
#                                                    texture_adjacency_directions=text_directions,
#                                                    window_width=window, distance=dist,
#                                                    texture_data_linked_to_spec=True, average=False,
#                                                    sat_name='images', accurate_pol=True, text_accurate_pol=False)
#
#         mask_samples.to_recolor({
#             'Field': 'lime',
#             'Forest': 'green'
#         })
#
#         mask_samples.to_save_samples_set('mask_train_selection_win_80_grad_50' + prefix, intermediate_data_directory)
#
#         mask2_samples = DataSamples.ClassSamplesSet(mask2_shape_address, spectral_shot_addresses, texture_shot_address,
#                                                    'CLASS',
#                                                     ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                                                     ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                                     texture_list=texture_list,
#                                                     texture_adjacency_directions=text_directions,
#                                                     window_width=window, distance=dist,
#                                                     texture_data_linked_to_spec=True, average=False,
#                                                     sat_name='images', accurate_pol=True, text_accurate_pol=False)
#
#         mask2_samples.to_recolor({
#             'Shrubs': 'y',
#             'Forest': 'green'
#         })
#
#         mask2_samples.to_save_samples_set('mask2_train_selection_win_80_grad_50' + prefix, intermediate_data_directory)
#
#         samples = DataSamples.ClassSamplesSet(shape_address, spectral_shot_addresses, texture_shot_address,
#                                               'CLASS',
#                                               ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                                               ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                               texture_list=texture_list,
#                                               texture_adjacency_directions=text_directions,
#                                               window_width=window, distance=dist,
#                                               texture_data_linked_to_spec=True, average=False,
#                                               sat_name='images',
#                                               exclude_shadow=False, offset=1, accurate_pol=True, text_accurate_pol=False)
#
#         samples.to_recolor({
#                             'O': 'green',
#                             #'P': 'cyan',
#                             #'S': 'dodgerblue',
#                             'A': 'red',
#                             'B': 'm',
#                             'L': 'orange',
#                             'LIN': 'magenta',
#                             #'LE': 'brown'
#         })
#
#         samples.to_save_samples_set('train_selection_win_80_grad_50 D' + prefix, intermediate_data_directory)
#         print(prefix)

# # Проект Бронницы (Sentinel)
#
# from numpy import pi
#
# import DataSamples
#
# if __name__ == "__main__":
#     mask_spectral_shot_addresses = ['D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.07.18)/GRANULE/L1C_T37UDB_A031707_20210718T085326/IMG_DATA/T37UDB_20210718T084601_B02.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.07.18)/GRANULE/L1C_T37UDB_A031707_20210718T085326/IMG_DATA/T37UDB_20210718T084601_B03.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.07.18)/GRANULE/L1C_T37UDB_A031707_20210718T085326/IMG_DATA/T37UDB_20210718T084601_B04.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.07.18)/GRANULE/L1C_T37UDB_A031707_20210718T085326/IMG_DATA/T37UDB_20210718T084601_B08.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.07.18)/GRANULE/L1C_T37UDB_A031707_20210718T085326/IMG_DATA/T37UDB_20210718T084601_B11.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.07.18)/GRANULE/L1C_T37UDB_A031707_20210718T085326/IMG_DATA/T37UDB_20210718T084601_B12.jp2'
#                                   ]
#     DC_spectral_shot_addresses = ['D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.11.23)/GRANULE/L1C_T37UDB_A008956_20181123T084251/IMG_DATA/T37UDB_20181123T084249_B02.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.11.23)/GRANULE/L1C_T37UDB_A008956_20181123T084251/IMG_DATA/T37UDB_20181123T084249_B03.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.11.23)/GRANULE/L1C_T37UDB_A008956_20181123T084251/IMG_DATA/T37UDB_20181123T084249_B04.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.11.23)/GRANULE/L1C_T37UDB_A008956_20181123T084251/IMG_DATA/T37UDB_20181123T084249_B08.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.11.23)/GRANULE/L1C_T37UDB_A008956_20181123T084251/IMG_DATA/T37UDB_20181123T084249_B11.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.11.23)/GRANULE/L1C_T37UDB_A008956_20181123T084251/IMG_DATA/T37UDB_20181123T084249_B12.jp2'
#                                   ]
#     DLE_spectral_shot_addresses = ['D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.10.14)/GRANULE/L1C_T37UDB_A008384_20181014T083913/IMG_DATA/T37UDB_20181014T083919_B02.jp2',
#                                    'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.10.14)/GRANULE/L1C_T37UDB_A008384_20181014T083913/IMG_DATA/T37UDB_20181014T083919_B03.jp2',
#                                    'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.10.14)/GRANULE/L1C_T37UDB_A008384_20181014T083913/IMG_DATA/T37UDB_20181014T083919_B04.jp2',
#                                    'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.10.14)/GRANULE/L1C_T37UDB_A008384_20181014T083913/IMG_DATA/T37UDB_20181014T083919_B08.jp2',
#                                    'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.10.14)/GRANULE/L1C_T37UDB_A008384_20181014T083913/IMG_DATA/T37UDB_20181014T083919_B11.jp2',
#                                    'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2018.10.14)/GRANULE/L1C_T37UDB_A008384_20181014T083913/IMG_DATA/T37UDB_20181014T083919_B12.jp2'
#                                    ]
#     C_spectral_shot_addresses = ['D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.09.13)/GRANULE/L1C_T37UDB_A032522_20210913T084310/IMG_DATA/T37UDB_20210913T083601_B02.jp2',
#                                  'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.09.13)/GRANULE/L1C_T37UDB_A032522_20210913T084310/IMG_DATA/T37UDB_20210913T083601_B03.jp2',
#                                  'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.09.13)/GRANULE/L1C_T37UDB_A032522_20210913T084310/IMG_DATA/T37UDB_20210913T083601_B04.jp2',
#                                  'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.09.13)/GRANULE/L1C_T37UDB_A032522_20210913T084310/IMG_DATA/T37UDB_20210913T083601_B08.jp2',
#                                  'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.09.13)/GRANULE/L1C_T37UDB_A032522_20210913T084310/IMG_DATA/T37UDB_20210913T083601_B11.jp2',
#                                  'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2021.09.13)/GRANULE/L1C_T37UDB_A032522_20210913T084310/IMG_DATA/T37UDB_20210913T083601_B12.jp2'
#                                  ]
#
#     D2_spectral_shot_addresses = [
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2017.07.29)/GRANULE/L1C_T37UDB_A010972_20170729T085325/IMG_DATA/T37UDB_20170729T085021_B02.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2017.07.29)/GRANULE/L1C_T37UDB_A010972_20170729T085325/IMG_DATA/T37UDB_20170729T085021_B03.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2017.07.29)/GRANULE/L1C_T37UDB_A010972_20170729T085325/IMG_DATA/T37UDB_20170729T085021_B04.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2017.07.29)/GRANULE/L1C_T37UDB_A010972_20170729T085325/IMG_DATA/T37UDB_20170729T085021_B08.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2017.07.29)/GRANULE/L1C_T37UDB_A010972_20170729T085325/IMG_DATA/T37UDB_20170729T085021_B11.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2017.07.29)/GRANULE/L1C_T37UDB_A010972_20170729T085325/IMG_DATA/T37UDB_20170729T085021_B12.jp2'
#                                   ]
#
#     D22_spectral_shot_addresses = ['D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2016.07.24)/IMG_DATA/S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB_B02.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2016.07.24)/IMG_DATA/S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB_B03.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2016.07.24)/IMG_DATA/S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB_B04.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2016.07.24)/IMG_DATA/S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB_B08.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2016.07.24)/IMG_DATA/S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB_B11.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2016.07.24)/IMG_DATA/S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37UDB_B12.jp2'
#                                   ]
#
#     texture_shot_address = 'D:/Проекты/Бронницы/Растровые данные/graded_texture_shot_50.tif'
#     mask_shape_address = 'D:/Проекты/Бронницы/Векторные данные/mask_train_selection.shp'
#     DC_shape_address = 'D:/Проекты/Бронницы/Векторные данные/train_selection_sentinel_DC.shp'
#     DLE_shape_address = 'D:/Проекты/Бронницы/Векторные данные/train_selection_sentinel_DLE.shp'
#     LE_shape_address = 'D:/Проекты/Бронницы/Векторные данные/train_selection_sentinel_LE.shp'
#     C_shape_address = 'D:/Проекты/Бронницы/Векторные данные/train_selection_sentinel_C.shp'
#     D2_shape_address = 'D:/Проекты/Бронницы/Векторные данные/train_selection_sentinel_D2.shp'
#
#     intermediate_data_directory = 'D:/Проекты/Бронницы/Промежуточные данные'
#     texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#
#     window = 80
#     text_directions = [0]
#     dist = 1
#
#     mask_samples = DataSamples.ClassSamplesSet(mask_shape_address, mask_spectral_shot_addresses, texture_shot_address,
#                                                'CLASS',
#                                                ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                                                ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                                texture_list=texture_list,
#                                                texture_adjacency_directions=text_directions,
#                                                window_width=window, distance=dist,
#                                                texture_data_linked_to_spec=True, average=False,
#                                                sat_name='images', accurate_pol=True, text_accurate_pol=False)
#
#     mask_samples.to_recolor({
#         'Field': 'lime',
#         'Forest': 'green',
#         'Build': 'gray',
#         'Water': 'blue'
#     })
#
#     mask_samples.to_save_samples_set('mask_train_selection_win_80_grad_50', intermediate_data_directory)
#
#     # mask2_samples = DataSamples.ClassSamplesSet(mask2_shape_address, spectral_shot_addresses, texture_shot_address,
#     #                                            'CLASS',
#     #                                             ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#     #                                             ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#     #                                             texture_list=texture_list,
#     #                                             texture_adjacency_directions=text_directions,
#     #                                             window_width=window, distance=dist,
#     #                                             texture_data_linked_to_spec=True, average=False,
#     #                                             sat_name='images', accurate_pol=True, text_accurate_pol=False)
#     #
#     # mask2_samples.to_recolor({
#     #     'Shrubs': 'y',
#     #     'Forest': 'green'
#     # })
#     #
#     # mask2_samples.to_save_samples_set('mask2_train_selection_win_80_grad_50', intermediate_data_directory)
#
#     DC_samples = DataSamples.ClassSamplesSet(DC_shape_address, DC_spectral_shot_addresses, texture_shot_address,
#                                               'CLASS',
#                                               ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                                               ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                               texture_list=texture_list,
#                                               texture_adjacency_directions=text_directions,
#                                               window_width=window, distance=dist,
#                                               texture_data_linked_to_spec=True, average=False,
#                                               sat_name='images',
#                                               exclude_shadow=False, offset=1, accurate_pol=True, text_accurate_pol=False)
#
#     DC_samples.to_recolor({
#         'O': 'green',
#         'A': 'green',
#         'B': 'green',
#         'L': 'green',
#         'LIN': 'green',
#         'LE': 'green',
#         'P': 'cyan',
#         'S': 'cyan'
#     })
#
#     DC_samples.to_save_samples_set('DC_train_selection_win_80_grad_50', intermediate_data_directory)
#
#     DLE_samples = DataSamples.ClassSamplesSet(DLE_shape_address, DLE_spectral_shot_addresses, texture_shot_address,
#                                               'CLASS',
#                                               ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                                               ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                               texture_list=texture_list,
#                                               texture_adjacency_directions=text_directions,
#                                               window_width=window, distance=dist,
#                                               texture_data_linked_to_spec=True, average=False,
#                                               sat_name='images',
#                                               exclude_shadow=False, offset=1, accurate_pol=True,
#                                               text_accurate_pol=False)
#
#     DLE_samples.to_recolor({
#         'O': 'green',
#         'A': 'green',
#         'B': 'green',
#         'L': 'green',
#         'LIN': 'green',
#         'LE': 'brown'
#     })
#
#     DLE_samples.to_save_samples_set('DLE_train_selection_win_80_grad_50', intermediate_data_directory)
#
#     LE_samples = DataSamples.ClassSamplesSet(LE_shape_address, DLE_spectral_shot_addresses, texture_shot_address,
#                                               'CLASS',
#                                               ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                                               ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                               texture_list=texture_list,
#                                               texture_adjacency_directions=text_directions,
#                                               window_width=window, distance=dist,
#                                               texture_data_linked_to_spec=True, average=False,
#                                               sat_name='images',
#                                               exclude_shadow=False, offset=1, accurate_pol=True,
#                                               text_accurate_pol=False)
#
#     LE_samples.to_recolor({
#         'LE': 'brown'
#     })
#
#     LE_samples.to_save_samples_set('LE_train_selection_win_80_grad_50', intermediate_data_directory)
#
#     C_samples = DataSamples.ClassSamplesSet(C_shape_address, C_spectral_shot_addresses, texture_shot_address,
#                                                 'CLASS',
#                                                 ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                                                 ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                                 texture_list=texture_list,
#                                                 texture_adjacency_directions=text_directions,
#                                                 window_width=window, distance=dist,
#                                                 texture_data_linked_to_spec=True, average=False,
#                                                 sat_name='images',
#                                                 exclude_shadow=False, offset=1, accurate_pol=True, text_accurate_pol=False)
#
#     C_samples.to_recolor({
#         'P': 'cyan',
#         'S': 'dodgerblue'
#     })
#
#     C_samples.to_save_samples_set('C_train_selection_win_80_grad_50', intermediate_data_directory)
#
#     D2_samples = DataSamples.ClassSamplesSet(D2_shape_address, D2_spectral_shot_addresses, texture_shot_address,
#                                                'CLASS',
#                                                ['B3', 'B4', 'B8', 'B11', 'B12'],
#                                                ['green', 'red', 'nir', 'swir1', 'swir2'],
#                                                texture_list=texture_list,
#                                                texture_adjacency_directions=text_directions,
#                                                window_width=window, distance=dist,
#                                                texture_data_linked_to_spec=True, average=False,
#                                                sat_name='images',
#                                                exclude_shadow=False, offset=1, accurate_pol=True,
#                                                text_accurate_pol=False)
#
#     D2_samples.to_recolor({
#         'O': 'green',
#         'A': 'red',
#         'B': 'blueviolet',
#         'L': 'orange',
#         'LIN': 'magenta'
#     })
#
#     D2_samples.to_save_samples_set('D2_train_selection_win_80_grad_50', intermediate_data_directory)
#
#     D22_samples = DataSamples.ClassSamplesSet(D2_shape_address, D22_spectral_shot_addresses, texture_shot_address,
#                                              'CLASS',
#                                              ['B3', 'B4', 'B8', 'B11', 'B12'],
#                                              ['green', 'red', 'nir', 'swir1', 'swir2'],
#                                              texture_list=texture_list,
#                                              texture_adjacency_directions=text_directions,
#                                              window_width=window, distance=dist,
#                                              texture_data_linked_to_spec=True, average=False,
#                                              sat_name='images',
#                                              exclude_shadow=False, offset=1, accurate_pol=True,
#                                              text_accurate_pol=False)
#
#     D22_samples.to_recolor({
#         'O': 'green',
#         'A': 'red',
#         'B': 'blueviolet',
#         'L': 'orange',
#         'LIN': 'magenta'
#     })
#
#     D22_samples.to_save_samples_set('D22_train_selection_win_80_grad_50', intermediate_data_directory)

# # Проект Бронницы (corection Sentinel)
#
# from numpy import pi
#
# import DataSamples
# import DataShot
#
# if __name__ == "__main__":
#     model = -1
#     F1_shot_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_Sentinel_win_80_grad_50_full_aerosol_(2019.06.04).file'
#     F2_shot_address = 'D:/Проекты/Бронницы/Промежуточные данные/Shot_Sentinel_win_80_grad_50_full_aerosol_(2019.06.06).file'
#     F1_shot = DataShot.to_load_data_shot(F1_shot_address)
#     F2_shot = DataShot.to_load_data_shot(F2_shot_address)
#     F1_shot.DOS_colebration('aerosol', model, start_point='min', show_correction_curve=False, show_start_point=False, map_band='red', neg_rule='calib')
#     F2_shot.DOS_colebration('aerosol', model, start_point='min', show_correction_curve=False, show_start_point=False, map_band='red', neg_rule='calib')
#
#     F1_spectral_shot_addresses = [
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2019.06.04)/GRANULE/L1C_T37UDB_A011716_20190604T085141/IMG_DATA/T37UDB_20190604T084609_B02.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2019.06.04)/GRANULE/L1C_T37UDB_A011716_20190604T085141/IMG_DATA/T37UDB_20190604T084609_B03.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2019.06.04)/GRANULE/L1C_T37UDB_A011716_20190604T085141/IMG_DATA/T37UDB_20190604T084609_B04.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2019.06.04)/GRANULE/L1C_T37UDB_A011716_20190604T085141/IMG_DATA/T37UDB_20190604T084609_B08.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2019.06.04)/GRANULE/L1C_T37UDB_A011716_20190604T085141/IMG_DATA/T37UDB_20190604T084609_B11.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2019.06.04)/GRANULE/L1C_T37UDB_A011716_20190604T085141/IMG_DATA/T37UDB_20190604T084609_B12.jp2'
#                                   ]
#
#     F2_spectral_shot_addresses = ['D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2019.06.06)/GRANULE/L1C_T37UDB_A020653_20190606T083602/IMG_DATA/T37UDB_20190606T083601_B02.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2019.06.06)/GRANULE/L1C_T37UDB_A020653_20190606T083602/IMG_DATA/T37UDB_20190606T083601_B03.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2019.06.06)/GRANULE/L1C_T37UDB_A020653_20190606T083602/IMG_DATA/T37UDB_20190606T083601_B04.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2019.06.06)/GRANULE/L1C_T37UDB_A020653_20190606T083602/IMG_DATA/T37UDB_20190606T083601_B08.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2019.06.06)/GRANULE/L1C_T37UDB_A020653_20190606T083602/IMG_DATA/T37UDB_20190606T083601_B11.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2019.06.06)/GRANULE/L1C_T37UDB_A020653_20190606T083602/IMG_DATA/T37UDB_20190606T083601_B12.jp2'
#                                   ]
#
#     texture_shot_address = 'D:/Проекты/Бронницы/Растровые данные/graded_texture_shot_50.tif'
#     mask_shape_address = 'D:/Проекты/Бронницы/Векторные данные/mask_train_selection.shp'
#     F_shape_address = 'D:/Проекты/Бронницы/Векторные данные/train_selection_sentinel.shp'
#
#     intermediate_data_directory = 'D:/Проекты/Бронницы/Промежуточные данные'
#     texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#
#     window = 80
#     text_directions = [0]
#     dist = 1
#
#     mask1_samples = DataSamples.ClassSamplesSet(mask_shape_address, F1_spectral_shot_addresses, texture_shot_address,
#                                                'CLASS',
#                                                ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                                                ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                                texture_list=texture_list,
#                                                texture_adjacency_directions=text_directions,
#                                                window_width=window, distance=dist,
#                                                texture_data_linked_to_spec=True, average=False,
#                                                sat_name='images', accurate_pol=True, text_accurate_pol=False,
#                                                shot=F1_shot)
#     mask1_samples.to_recolor({
#         'Field': 'lime',
#         'Forest': 'green',
#         'Build': 'gray',
#         'Water': 'blue'
#     })
#
#     mask1_samples.to_save_samples_set('mask1_train_selection_win_80_grad_50', intermediate_data_directory)
#
#     mask2_samples = DataSamples.ClassSamplesSet(mask_shape_address, F2_spectral_shot_addresses, texture_shot_address,
#                                                 'CLASS',
#                                                 ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                                                 ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                                 texture_list=texture_list,
#                                                 texture_adjacency_directions=text_directions,
#                                                 window_width=window, distance=dist,
#                                                 texture_data_linked_to_spec=True, average=False,
#                                                 sat_name='images', accurate_pol=True, text_accurate_pol=False,
#                                                 shot=F2_shot)
#
#     mask2_samples.to_recolor({
#         'Field': 'lime',
#         'Forest': 'green',
#         'Build': 'gray',
#         'Water': 'blue'
#     })
#
#     mask2_samples.to_save_samples_set('mask2_train_selection_win_80_grad_50', intermediate_data_directory)
#
#     F1_samples = DataSamples.ClassSamplesSet(F_shape_address, F1_spectral_shot_addresses, texture_shot_address,
#                                               'CLASS',
#                                               ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                                               ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                               texture_list=texture_list,
#                                               texture_adjacency_directions=text_directions,
#                                               window_width=window, distance=dist,
#                                               texture_data_linked_to_spec=True, average=False,
#                                               sat_name='images',
#                                               exclude_shadow=False, offset=1, accurate_pol=True, text_accurate_pol=False,
#                                               shot=F1_shot)
#
#     F1_samples.to_recolor({
#         'O': 'green',
#         'A': 'red',
#         'B': 'blueviolet',
#         'L': 'orange',
#         'LIN': 'magenta',
#         'LE': 'brown',
#         'P': 'cyan',
#         'S': 'dodgerblue'
#     })
#
#     F1_samples.to_save_samples_set('F1_train_selection_win_80_grad_50', intermediate_data_directory)
#
#     F2_samples = DataSamples.ClassSamplesSet(F_shape_address, F2_spectral_shot_addresses, texture_shot_address,
#                                              'CLASS',
#                                              ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                                              ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                              texture_list=texture_list,
#                                              texture_adjacency_directions=text_directions,
#                                              window_width=window, distance=dist,
#                                              texture_data_linked_to_spec=True, average=False,
#                                              sat_name='images',
#                                              exclude_shadow=False, offset=1, accurate_pol=True, text_accurate_pol=False,
#                                              shot=F2_shot)
#
#     F2_samples.to_recolor({
#         'O': 'green',
#         'A': 'red',
#         'B': 'blueviolet',
#         'L': 'orange',
#         'LIN': 'magenta',
#         'LE': 'brown',
#         'P': 'cyan',
#         'S': 'dodgerblue'
#     })
#
#     F2_samples.to_save_samples_set('F2_train_selection_win_80_grad_50', intermediate_data_directory)

# # Проект Бронницы (Landsat 8)
#
# from numpy import pi
#
# import DataSamples
#
# if __name__ == "__main__":
#     spectral_shot_address = 'D:/Проекты/Бронницы/Растровые данные/Landsat 8 (2014.07.28)/LC08_L2SP_177021_20140728_20200911_02_T1_MTL.txt'
#     texture_shot_address = 'D:/Проекты/Бронницы/Растровые данные/graded_texture_shot_50.tif'
#     shape_address = 'D:/Проекты/Бронницы/Векторные данные/train_selection_resurs.shp'
#     mask_shape_address = 'D:/Проекты/Бронницы/Векторные данные/mask_train_selection_resurs.shp'
#     intermediate_data_directory = 'D:/Проекты/Бронницы/Промежуточные данные'
#     texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#     #texture_list = ['Autocorrelation']
#
#     window = 80
#     text_directions = [0]
#     dist = 1
#
#     mask_samples = DataSamples.ClassSamplesSet(mask_shape_address, spectral_shot_address, texture_shot_address,
#                                                'CLASS',
#                                                ['BAND_2', 'BAND_3', 'BAND_4', 'BAND_5', 'BAND_6', 'BAND_7'],
#                                                ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                                texture_list=texture_list,
#                                                texture_adjacency_directions=text_directions,
#                                                window_width=window, distance=dist,
#                                                texture_data_linked_to_spec=True, average=False,
#                                                sat_name='Landsat_8')
#
#     mask_samples.to_recolor({
#         #'Water': 'blue',
#         'Field': 'lime',
#         'Forest': 'green',
#         #'Build': 'gray'
#     })
#
#     mask_samples.to_save_samples_set('mask_train_selection_win_80_grad_50', intermediate_data_directory)
#
#     samples = DataSamples.ClassSamplesSet(shape_address, spectral_shot_address, texture_shot_address,
#                                           'CLASS',
#                                           ['BAND_2', 'BAND_3', 'BAND_4', 'BAND_5', 'BAND_6', 'BAND_7'],
#                                           ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                           texture_list=texture_list,
#                                           texture_adjacency_directions=text_directions,
#                                           window_width=window, distance=dist,
#                                           texture_data_linked_to_spec=True, average=False,
#                                           sat_name='Landsat_8',
#                                           exclude_shadow=False, offset=1)
#
#     samples.to_recolor({
#                         'O': 'green',
#                         'P': 'cyan',
#                         'S': 'dodgerblue',
#                         'A': 'red',
#                         'B': 'm',
#                         'L': 'orange'
#                         })

#     samples.to_save_samples_set('train_selection_win_80_grad_50', intermediate_data_directory)

    # shape_address = 'D:/Проекты/Бронницы/Векторные данные/Бронницы_wgs_ацо.shp'
    #
    # samples = DataSamples.ClassSamplesSet(shape_address, spectral_shot_address, texture_shot_address,
    #                                       'TYPE',
    #                                       ['BAND_B', 'BAND_G', 'BAND_Y', 'BAND_R', 'BAND_RE', 'BAND_N', 'BAND_N2'],
    #                                       ['blue', 'green', 'yellow', 'red', 'red_edge', 'nir', 'nir2'],
    #                                       #['BAND_B'],
    #                                       #['blue'],
    #                                       texture_list=texture_list,
    #                                       texture_adjacency_directions=text_directions,
    #                                       window_width=window, distance=dist,
    #                                       texture_data_linked_to_spec=True, average=False,
    #                                       sat_name='WorldView_2')
    #
    # samples.to_save_samples_set('test_selection_win_80_grad_50', intermediate_data_directory)
    #
    # samples.to_recolor({
    #                     'D': 'green',
    #                     'C': 'cyan'
    #                     })

# # Создание cэмплов для задачи определения полноты для участка Валуйского лесничества
#
# from numpy import pi
#
# import DataSamples
#
# if __name__ == "__main__":
#    texture_shot_address = 'D:/Проекты/Структурные индексы/Растровые данные/p_graded_shot_100.tif'
#    shape_address = 'D:/Проекты/Структурные индексы/Векторные данные/train_samples_reg.shp'
#    intermediate_data_directory = 'D:/Проекты/Структурные индексы/Промежуточные данные'
#    texture_list = ['SDGL', 'Contrast', 'Entropy']
#
#    window = 64
#    text_directions = [0]
#    dist = 1
#
#    samples = DataSamples.ClassSamplesSet(shape_address, None, texture_shot_address,
#                                          'SKAL1',
#                                          None,
#                                          None,
#                                          texture_list=texture_list, texture_adjacency_directions=text_directions,
#                                          window_width=window, distance=dist,
#                                          texture_data_linked_to_spec=False, average=False, samples_type='regression')
#
#    samples.to_recolor([(0.0, 'white'),
#                        (0.4, 'red'),
#                        (0.5, 'orange'),
#                        (0.6, 'yellow'),
#                        (0.7, 'lime'),
#                        (0.8, 'cyan'),
#                        (0.9, 'blue'),
#                        (1.0, 'purple')
#                        ])
#    # samples.to_recolor({'0.0': 'white',
#    #                     #'0.4': 'red',
#    #                     #'0.5': 'orange',
#    #                     #'0.6': 'yellow',
#    #                     '0.7': 'lime',
#    #                     #'0.8': 'cyan',
#    #                     #'0.9': 'blue'
#    #                     '1.0': 'purple'
#    #                     })
#    # samples.to_recolor({'0': 'yellow',
#    #                     '1': 'lime'})
#
#    samples.to_save_samples_set('p_train_selection_100', intermediate_data_directory)
#
#    shape_address = 'D:/Проекты/Структурные индексы/Векторные данные/test_samples_reg.shp'
#    intermediate_data_directory = 'D:/Проекты/Структурные индексы/Промежуточные данные'
#
#    window = 64
#    text_directions = [0]
#    dist = 1
#
#    samples = DataSamples.ClassSamplesSet(shape_address, None, texture_shot_address,
#                                          'SKAL1',
#                                          None,
#                                          None,
#                                          texture_list=texture_list, texture_adjacency_directions=text_directions,
#                                          window_width=window, distance=dist,
#                                          texture_data_linked_to_spec=False, average=False, samples_type='regression')
#
#    samples.to_recolor([(0.0, 'white'),
#                        (0.4, 'red'),
#                        (0.5, 'orange'),
#                        (0.6, 'yellow'),
#                        (0.7, 'lime'),
#                        (0.8, 'cyan'),
#                        (0.9, 'blue'),
#                        (1.0, 'purple')
#                        ])
#    # samples.to_recolor({'0.0': 'white',
#    #                     #'0.4': 'red',
#    #                     #'0.5': 'orange',
#    #                     #'0.6': 'yellow',
#    #                     '0.7': 'lime',
#    #                     #'0.8': 'cyan',
#    #                     #'0.9': 'blue'
#    #                     '1.0': 'purple'
#    #                     })
#    # samples.to_recolor({'0': 'yellow',
#    #                     '1': 'lime'})
#
#    samples.to_save_samples_set('p_test_selection_100', intermediate_data_directory)

# Проект Сербия (Sentinel)

from numpy import pi

import DataSamples

if __name__ == "__main__":
    spectral_shot_addresses = [
                               'D:/Проекты/Велики Столак/Raster/qgis_DOS_Sentinel 2021.05.04/RT_T34TCP_20210504T094031_B02.tif',
                               'D:/Проекты/Велики Столак/Raster/qgis_DOS_Sentinel 2021.05.04/RT_T34TCP_20210504T094031_B03.tif',
                               'D:/Проекты/Велики Столак/Raster/qgis_DOS_Sentinel 2021.05.04/RT_T34TCP_20210504T094031_B04.tif',
                               'D:/Проекты/Велики Столак/Raster/qgis_DOS_Sentinel 2021.05.04/RT_T34TCP_20210504T094031_B08.tif',
                               'D:/Проекты/Велики Столак/Raster/qgis_DOS_Sentinel 2021.05.04/RT_T34TCP_20210504T094031_B11.tif',
                               'D:/Проекты/Велики Столак/Raster/qgis_DOS_Sentinel 2021.05.04/RT_T34TCP_20210504T094031_B12.tif'
                               ]

    shape_address = 'D:/Проекты/Велики Столак/Vector/Border_2.shp'

    intermediate_data_directory = 'D:/Проекты/Велики Столак/inter_data'

    samples = DataSamples.ClassSamplesSet(shape_address, spectral_shot_addresses, None,
                                               'id',
                                               ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
                                               ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
                                               sat_name='images', accurate_pol=True)

    samples.to_save_samples_set('qgis_DOS_selection', intermediate_data_directory)

# # Создание cэмплов для задачи определения полноты для участка Валуйского лесничества
#
# from numpy import pi
#
# import DataSamples
#
# if __name__ == "__main__":
#     texture_shot_address = 'D:/Проекты/Структурные индексы/Растровые данные/graded_shot_50.tif'
#     shape_address = 'D:/Проекты/Структурные индексы/Векторные данные/train_samples_reg.shp'
#     intermediate_data_directory = 'D:/Проекты/Структурные индексы/Промежуточные данные'
#     texture_list = ['SDGL', 'Contrast', 'Entropy']
#
#     window = 16
#     text_directions = [0]
#     dist = 1
#
#     samples = DataSamples.ClassSamplesSet(shape_address, None, texture_shot_address,
#                                           'SKAL1',
#                                           None,
#                                           None,
#                                           texture_list=texture_list, texture_adjacency_directions=text_directions,
#                                           window_width=window, distance=dist,
#                                           texture_data_linked_to_spec=False, average=False, samples_type='regression')
#
#     samples.to_recolor([(0.0, 'white'),
#                         (0.4, 'red'),
#                         (0.5, 'orange'),
#                         (0.6, 'yellow'),
#                         (0.7, 'lime'),
#                         (0.8, 'cyan'),
#                         (0.9, 'blue'),
#                         (1.0, 'purple')
#                         ])
#     # samples.to_recolor({'0.0': 'white',
#     #                     #'0.4': 'red',
#     #                     #'0.5': 'orange',
#     #                     #'0.6': 'yellow',
#     #                     '0.7': 'lime',
#     #                     #'0.8': 'cyan',
#     #                     #'0.9': 'blue'
#     #                     '1.0': 'purple'
#     #                     })
#     # samples.to_recolor({'0': 'yellow',
#     #                     '1': 'lime'})
#
#     samples.to_save_samples_set('train_selection', intermediate_data_directory)
#
#     shape_address = 'D:/Проекты/Структурные индексы/Векторные данные/test_samples_reg.shp'
#
#     samples = DataSamples.ClassSamplesSet(shape_address, None, texture_shot_address,
#                                           'SKAL1',
#                                           None,
#                                           None,
#                                           texture_list=texture_list, texture_adjacency_directions=text_directions,
#                                           window_width=window, distance=dist,
#                                           texture_data_linked_to_spec=False, average=False, samples_type='regression')
#
#     samples.to_recolor([(0.0, 'white'),
#                         (0.4, 'red'),
#                         (0.5, 'orange'),
#                         (0.6, 'yellow'),
#                         (0.7, 'lime'),
#                         (0.8, 'cyan'),
#                         (0.9, 'blue'),
#                         (1.0, 'purple')
#                         ])
#     # samples.to_recolor({'0.0': 'white',
#     #                     '0.4': 'red',
#     #                     '0.5': 'orange',
#     #                     '0.6': 'yellow',
#     #                     '0.7': 'lime',
#     #                     '0.8': 'cyan',
#     #                     '0.9': 'blue',
#     #                     '1.0': 'purple'})
#     # #samples.to_recolor({'0': 'yellow',
#     # #                    '1': 'lime'})
#
#     samples.to_save_samples_set('test_selection', intermediate_data_directory)

#from numpy import pi
#
#import DataSamples
#
#if __name__ == "__main__":
#    spectral_shot_addresses = ['D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/'
#                               'Общие данные/Rastrs/Спектр/Sentinel 2/GRANULE/IMG_DATA/'
#                               'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37VCD_B02.jp2',
#                               'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/'
#                               'Общие данные/Rastrs/Спектр/Sentinel 2/GRANULE/IMG_DATA/'
#                               'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37VCD_B03.jp2',
#                               'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/'
#                               'Общие данные/Rastrs/Спектр/Sentinel 2/GRANULE/IMG_DATA/'
#                               'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37VCD_B04.jp2',
#                               'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/'
#                               'Общие данные/Rastrs/Спектр/Sentinel 2/GRANULE/IMG_DATA/'
#                               'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37VCD_B08.jp2',
#                               'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/'
#                               'Общие данные/Rastrs/Спектр/Sentinel 2/GRANULE/IMG_DATA/'
#                               'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37VCD_B11.jp2',
#                               'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/'
#                               'Общие данные/Rastrs/Спектр/Sentinel 2/GRANULE/IMG_DATA/'                                'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37VCD_B12.jp2',
#                                ]
#    #texture_shot_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во/' \
#    #                            'Общие данные/Rastrs/Текстурные данные/Запад/056009302010_01_P002_PAN/' \
#    #                            '16JUN25085338-P2AS-056009302010_01_P002.TIF'
#    #texture_shot_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во/' \
#    #                            'Общие данные/Rastrs/Текстурные данные/Восток/056009302010_01_P001_PAN/' \
#    #                            '16JUN25085327-P2AS-056009302010_01_P001.TIF'
#    texture_shot_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                                'Общие данные/Rastrs/Текстурные данные/Запад/Градуированные данные/high res 100 (west).tif'
##    texture_shot_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                                'Общие данные/Rastrs/Текстурные данные/Восток/Градуированные данные/high res 100 (east).tif'
#    samples_shape_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                                 'Классификации/Классификация леса на лиственный и хвойный/Shapes/test_selection (west).shp'
#    samples_shape_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                                 'Классификации/Классификация леса на лиственный и хвойный/Shapes/test_selection (east).shp'
#    intermediate_data_directory = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                                  'Классификации/Классификация леса на лиственный и хвойный/Промежуточные результаты'
#    texture_list = []
#    # texture_list = ['Contrast']
#    window = 80
#    text_directions = [0, 3 * pi / 4]
#    dist = 1
#    grad_count = None
#
#    samples = DataSamples.ClassSamplesSet(samples_shape_address_west, spectral_shot_addresses, texture_shot_address_west,
#                                          'CLASS',
#                                          ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                                          ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                          texture_list=texture_list, texture_adjacency_directions=text_directions,
#                                          window_width=window, grad_count=grad_count, distance=dist,
#                                          texture_data_linked_to_spec=True, average=False, sat_name='images')
#    samples.to_add_samples(samples_shape_address_east, spectral_shot_addresses, texture_shot_address_east,
#                           'CLASS',
#                           ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                           ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                           texture_list=texture_list, texture_adjacency_directions=text_directions,
#                           window_width=window, grad_count=grad_count, distance=dist,
#                           texture_data_linked_to_spec=True, average=False, sat_name='images')
#
#    samples.to_recolor({'Water': 'blue',
#                        'Grass': 'limegreen',
#                        'Sand': 'y',
#                        'Town': 'darkorange',
#                        'Deciduous': 'green',
#                        'Coniferous': 'deepskyblue'})
#
#    samples.to_save_samples_set('samples', intermediate_data_directory)

#WorldView2
# from numpy import pi
#
# import DataSamples
#
# if __name__ == "__main__":
#     spectral_shot_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (WorldView 2)/' \
#                                  'Общие данные/Rastrs/Спектр/WorldView 2/056009302010_01_P002_MUL/16JUN25085338-M2AS-056009302010-01-P002.XML'
#     spectral_shot_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (WorldView 2)/' \
#                                  'Общие данные/Rastrs/Спектр/WorldView 2/056009302010_01_P001_MUL/16JUN25085327-M2AS-056009302010-01-P001.XML'
#     #texture_shot_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во/' \
#     #                            'Общие данные/Rastrs/Текстурные данные/Запад/056009302010_01_P002_PAN/' \
#     #                            '16JUN25085338-P2AS-056009302010_01_P002.TIF'
#     #texture_shot_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во/' \
#     #                            'Общие данные/Rastrs/Текстурные данные/Восток/056009302010_01_P001_PAN/' \
#     #                            '16JUN25085327-P2AS-056009302010_01_P001.TIF'
#     texture_shot_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                                 'Общие данные/Rastrs/Текстурные данные/Запад/Градуированные данные/high res 100 (west).tif'
#     texture_shot_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                                 'Общие данные/Rastrs/Текстурные данные/Восток/Градуированные данные/high res 100 (east).tif'
#     samples_shape_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                                  'Классификации/Классификация леса на лиственный и хвойный/Samples/Запад/train_selection (west).shp'
#     samples_shape_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                                  'Классификации/Классификация леса на лиственный и хвойный/Samples/Восток/train_selection (east).shp'
#     intermediate_data_directory = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                                   'Классификации/Классификация леса на лиственный и хвойный/Промежуточные результаты'
#
#     # texture_list = []
#     texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#     window = 80
#     text_directions = [0, 3 * pi / 4]
#     dist = 1
#     grad_count = None
#
#     samples = DataSamples.ClassSamplesSet(samples_shape_address_west, spectral_shot_address_west,
#                                           texture_shot_address_west,
#                                           'CLASS2',
#                                           ['BAND_B', 'BAND_G', 'BAND_R', 'BAND_N', 'BAND_N2'],
#                                           ['blue', 'green', 'red', 'nir', 'nir2'],
#                                           sat_name='WorldView_2',
#                                           texture_list=texture_list, texture_adjacency_directions=text_directions,
#                                           window_width=window, grad_count=grad_count, distance=dist,
#                                           texture_data_linked_to_spec=True, average=False)
#     samples.to_add_samples(samples_shape_address_east, spectral_shot_address_east, texture_shot_address_east,
#                            'CLASS2',
#                            ['BAND_B', 'BAND_G', 'BAND_R', 'BAND_N', 'BAND_N2'],
#                            ['blue', 'green', 'red', 'nir', 'nir2'],
#                            sat_name='WorldView_2',
#                            texture_list=texture_list, texture_adjacency_directions=text_directions,
#                            window_width=window, grad_count=grad_count, distance=dist,
#                            texture_data_linked_to_spec=True, average=False)
#
#     samples.to_recolor({'Water': 'blue',
#                         'Grass': 'limegreen',
#                         'Sand': 'y',
#                         'Town': 'darkorange',
#                         'D': 'green',
#                         'C': 'deepskyblue'})
#
#     samples.to_save_samples_set('samples', intermediate_data_directory)


# Sentinel2
from numpy import pi

import DataSamples

#if __name__ == "__main__":
#   spectral_shot_addresses = [
#                             'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/'
#                             'Общие данные/Rastrs/Спектр/Sentinel 2/GRANULE/IMG_DATA/'
#                             'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37VCD_B02.jp2',
#                             'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/'
#                             'Общие данные/Rastrs/Спектр/Sentinel 2/GRANULE/IMG_DATA/'
#                             'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37VCD_B03.jp2',
#                             'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/'
#                             'Общие данные/Rastrs/Спектр/Sentinel 2/GRANULE/IMG_DATA/'
#                             'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37VCD_B04.jp2',
#                             'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/'
#                             'Общие данные/Rastrs/Спектр/Sentinel 2/GRANULE/IMG_DATA/'
#                             'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37VCD_B08.jp2',
#                             'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/'
#                             'Общие данные/Rastrs/Спектр/Sentinel 2/GRANULE/IMG_DATA/'
#                             'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37VCD_B11.jp2',
#                             'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/'
#                             'Общие данные/Rastrs/Спектр/Sentinel 2/GRANULE/IMG_DATA/'
#                             'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37VCD_B12.jp2'
#                             ]
#   #texture_shot_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во/' \
#   #                            'Общие данные/Rastrs/Текстурные данные/Запад/056009302010_01_P002_PAN/' \
#   #                            '16JUN25085338-P2AS-056009302010_01_P002.TIF'
#   #texture_shot_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во/' \
#   #                            'Общие данные/Rastrs/Текстурные данные/Восток/056009302010_01_P001_PAN/' \
#   #                            '16JUN25085327-P2AS-056009302010_01_P001.TIF'
#   texture_shot_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                               'Общие данные/Rastrs/Текстурные данные/Запад/Градуированные данные/high res 100 (west).tif'
#   texture_shot_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                               'Общие данные/Rastrs/Текстурные данные/Восток/Градуированные данные/high res 100 (east).tif'
#   samples_shape_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                                'Классификации/Классификация леса на лиственный и хвойный/Samples/Запад/train_selection (west).shp'
#   samples_shape_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                                'Классификации/Классификация леса на лиственный и хвойный/Samples/Восток/train_selection (east).shp'
#   intermediate_data_directory = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                                 'Классификации/Классификация леса на лиственный и хвойный/Промежуточные результаты'
#
#   # texture_list = []
#   texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#   window = 80
#   text_directions = [0, 3 * pi / 4]
#   dist = 1
#   grad_count = None
#
#   samples = DataSamples.ClassSamplesSet(samples_shape_address_west, spectral_shot_addresses,
#                                         texture_shot_address_west,
#                                         'CLASS2',
#                                         ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                                         ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                         sat_name='images',
#                                         texture_list=texture_list, texture_adjacency_directions=text_directions,
#                                         window_width=window, grad_count=grad_count, distance=dist,
#                                         texture_data_linked_to_spec=True, average=False)
#   samples.to_add_samples(samples_shape_address_east, spectral_shot_addresses, texture_shot_address_east,
#                          'CLASS2',
#                          ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                          ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                          sat_name='images',
#                          texture_list=texture_list, texture_adjacency_directions=text_directions,
#                          window_width=window, grad_count=grad_count, distance=dist,
#                          texture_data_linked_to_spec=True, average=False)
#
#   samples.to_recolor({'Water': 'blue',
#                       'Grass': 'limegreen',
#                       'Sand': 'y',
#                       'Town': 'darkorange',
#                       'D': 'green',
#                       'C': 'deepskyblue'})
#
#   samples.to_save_samples_set('samples', intermediate_data_directory)
#
# Landsat8
# from numpy import pi
#
# import DataSamples
#
# if __name__ == "__main__":
#    spectral_shot_address = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                            'Общие данные/Rastrs/Спектр/Landsat 8/2016.06.29/LC08_L1TP_179020_20160629_20170323_01_T1_MTL.txt'
#    #texture_shot_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во/' \
#    #                            'Общие данные/Rastrs/Текстурные данные/Запад/056009302010_01_P002_PAN/' \
#    #                            '16JUN25085338-P2AS-056009302010_01_P002.TIF'
#    #texture_shot_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во/' \
#    #                            'Общие данные/Rastrs/Текстурные данные/Восток/056009302010_01_P001_PAN/' \
#    #                            '16JUN25085327-P2AS-056009302010_01_P001.TIF'
#    texture_shot_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                                'Общие данные/Rastrs/Текстурные данные/Запад/Градуированные данные/high res 100 (west).tif'
#    texture_shot_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                                'Общие данные/Rastrs/Текстурные данные/Восток/Градуированные данные/high res 100 (east).tif'
#    samples_shape_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                                 'Классификации/Классификация леса на лиственный и хвойный/Samples/Запад/train_selection (west).shp'
#    samples_shape_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                                 'Классификации/Классификация леса на лиственный и хвойный/Samples/Восток/train_selection (east).shp'
#    intermediate_data_directory = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                                  'Классификации/Классификация леса на лиственный и хвойный/Промежуточные результаты'
#
#    # texture_list = []
#    texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#    window = 80
#    text_directions = [0, 3 * pi / 4]
#    dist = 1
#    grad_count = None
#
#    samples = DataSamples.ClassSamplesSet(samples_shape_address_west, spectral_shot_address,
#                                          texture_shot_address_west,
#                                          'CLASS2',
#                                          ['BAND_3', 'BAND_4', 'BAND_6', 'BAND_7'],
#                                          ['green', 'red', 'swir1', 'swir2'],
#                                          sat_name='Landsat_8',
#                                          texture_list=texture_list, texture_adjacency_directions=text_directions,
#                                          window_width=window, grad_count=grad_count, distance=dist,
#                                          texture_data_linked_to_spec=True, average=False)
#    samples.to_add_samples(samples_shape_address_east, spectral_shot_address, texture_shot_address_east,
#                           'CLASS2',
#                           ['BAND_3', 'BAND_4', 'BAND_6', 'BAND_7'],
#                           ['green', 'red', 'swir1', 'swir2'],
#                           sat_name='Landsat_8',
#                           texture_list=texture_list, texture_adjacency_directions=text_directions,
#                           window_width=window, grad_count=grad_count, distance=dist,
#                           texture_data_linked_to_spec=True, average=False)
#
#    samples.to_recolor({'Water': 'blue',
#                        'Grass': 'limegreen',
#                        'Sand': 'y',
#                        'Town': 'darkorange',
#                        'D': 'green',
#                        'C': 'deepskyblue'})
#
#    samples.to_save_samples_set('samples', intermediate_data_directory)