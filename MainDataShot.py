# # Тестирование адаптивного порога
# import numpy as np
# from numpy import pi
#
# import DataShot
#
# if __name__ == "__main__":
#    spectral_shot_address = 'D:/Проекты/Бронницы/Растровые данные/058041098010_01/058041098010_01_P001_MUL/' \
#                            '11JUL28090720-M2AS-058041098010_01_P001.XML'
#    texture_shot_address = 'D:/Проекты/Бронницы/Растровые данные/058041098010_01/058041098010_01_P001_PAN/' \
#                           '11JUL28090720-P2AS-058041098010_01_P001.TIF'
#    border_shape_address = 'D:/Проекты/Бронницы/Векторные данные/test.shp'
#    intermediate_data_directory = 'D:/Проекты/Бронницы/Промежуточные данные'
#
#    window = 80
#    distances = [1]
#    gradations = [100]
#    directions = [0]
#
#    texture_list = None
#
#    shot_name = 'WorldView_test_no_shadow'
#    shot = DataShot.DataShot(shot_name, border_shape_address, False)
#    shot.to_add_spectral_data_from_worldview2(['BAND_B', 'BAND_G', 'BAND_Y', 'BAND_R', 'BAND_RE', 'BAND_N', 'BAND_N2'],
#                                              ['blue', 'green', 'yellow', 'red', 'red_edge', 'nir', 'nir2'],
#                                              spectral_shot_address)
#    #shot.to_save_image_as_geotiff(shot.ndvi, shot.spec_geo_trans, shot.spec_projection_ref, 'ndvi', intermediate_data_directory)
#    shot.to_add_texture_data(texture_shot_address,
#                             directions=directions, window_width=window, distance=distances, grad_count=gradations,
#                             texture_data_linked_to_spec=True,
#                             to_save_clipped_image_like_high_res=False)
#
#    #shot.remove_shadows(offset=0.5)
#
#    shot.to_save_data_shot(shot_name, intermediate_data_directory)


# # Проект Бронницы (WorldView)
# import numpy as np
# from numpy import pi
#
# import DataShot
#
# if __name__ == "__main__":
#    spectral_shot_address = 'D:/Проекты/Бронницы/Растровые данные/058041098010_01/058041098010_01_P001_MUL/' \
#                            '11JUL28090720-M2AS-058041098010_01_P001.XML'
#    texture_shot_address = 'D:/Проекты/Бронницы/Растровые данные/058041098010_01/058041098010_01_P001_PAN/' \
#                           '11JUL28090720-P2AS-058041098010_01_P001.TIF'
#    border_shape_address = 'D:/Проекты/Бронницы/Векторные данные/Border_resurs.shp'
#    intermediate_data_directory = 'D:/Проекты/Бронницы/Промежуточные данные'
#
#    window = 80
#    distances = [1]
#    gradations = [50]
#    directions = [0]
#
#    texture_list = None
#
#    shot_name = 'Shot_WorldView_2'
#    shot = DataShot.DataShot(shot_name, border_shape_address, False)
#    shot.to_add_spectral_data_from_worldview2(['BAND_B', 'BAND_G', 'BAND_Y', 'BAND_R', 'BAND_RE', 'BAND_N', 'BAND_N2'],
#                                              ['blue', 'green', 'yellow', 'red', 'red_edge', 'nir', 'nir2'],
#                                              spectral_shot_address)
#    #shot.to_make_ndvi('red', 'nir')
#    #shot.to_save_image_as_geotiff(shot.ndvi, shot.spec_geo_trans, shot.spec_projection_ref, 'ndvi', intermediate_data_directory)
#    shot.to_add_texture_data(texture_shot_address,
#                             directions=directions, window_width=window, distance=distances, grad_count=gradations,
#                             texture_data_linked_to_spec=True,
#                             to_save_clipped_image_like_high_res=False)
#    shot.to_save_data_shot(shot_name, intermediate_data_directory)

# # Проект Бронницы (Sentinel)
# import numpy as np
# from numpy import pi
#
# import DataShot
#
# if __name__ == "__main__":
#     spectral_shot_addresses = ['D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2017.07.29)/GRANULE/L1C_T37UDB_A010972_20170729T085325/IMG_DATA/T37UDB_20170729T085021_B02.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2017.07.29)/GRANULE/L1C_T37UDB_A010972_20170729T085325/IMG_DATA/T37UDB_20170729T085021_B03.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2017.07.29)/GRANULE/L1C_T37UDB_A010972_20170729T085325/IMG_DATA/T37UDB_20170729T085021_B04.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2017.07.29)/GRANULE/L1C_T37UDB_A010972_20170729T085325/IMG_DATA/T37UDB_20170729T085021_B08.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2017.07.29)/GRANULE/L1C_T37UDB_A010972_20170729T085325/IMG_DATA/T37UDB_20170729T085021_B11.jp2',
#                                   'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2017.07.29)/GRANULE/L1C_T37UDB_A010972_20170729T085325/IMG_DATA/T37UDB_20170729T085021_B12.jp2'
#                                   ]
#     texture_shot_address = 'D:/Проекты/Бронницы/Растровые данные/058041098010_01/058041098010_01_P001_PAN/' \
#                            '11JUL28090720-P2AS-058041098010_01_P001.TIF'
#     border_shape_address = 'D:/Проекты/Бронницы/Векторные данные/Border.shp'
#     intermediate_data_directory = 'D:/Проекты/Бронницы/Промежуточные данные'
#
#     window = 80
#     distances = [1]
#     gradations = [50]
#     directions = [0]
#
#     texture_list = None
#
#     shot_name = 'Shot_Sentinel_win_80_grad_50_full_(2017.07.29)'
#
#     upper_wavelengths = [0.523, 0.578, 0.680, 0.903, 1.655, 2.280]
#     lower_wavelengths = [0.458, 0.543, 0.650, 0.788, 1.565, 2.100]
#
#     # import gdal
#     # from DataSamples import to_clip_shot
#     # rast = gdal.Open(texture_shot_address)
#     # clipped_texture_im, new_texture_geo_trans = to_clip_shot(rast,
#     #                                                          polygon,
#     #                                                          texture_geo_trans,f
#     #                                                          texture_projection_ref,
#     #                                                          rectangle_shape=rectangle_shape,
#     #                                                          origin=origin,
#     #                                                          grad_count=grad_count)
#
#     shot = DataShot.DataShot(shot_name, border_shape_address, False)
#     shot.to_add_spectral_data_from_images(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                                         ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                         spectral_shot_addresses,
#                                         [None, None, None, None, None, None], upper_wavelengths, lower_wavelengths,
#                                         [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0])
#
#     # shot.to_make_ndvi('red', 'nir')
#     # shot.to_save_image_as_geotiff(shot.ndvi, shot.spec_geo_trans, shot.spec_projection_ref, 'ndvi',
#     #                               'D:/Проекты/Бронницы/Растровые данные')
#     # shot.to_make_pseudo_color_image('pseudo_color_image', 'D:/Проекты/Бронницы/Растровые данные', 'red', 'green', 'nir')
#     shot.to_make_rgb('red', 'green', 'blue',
#                      red_band_add=0, green_band_add=0, blue_band_add=0,
#                      red_band_mult=1, green_band_mult=1, blue_band_mult=1)
#     shot.to_save_image_as_geotiff(shot.rgb_image, shot.spec_geo_trans, shot.spec_projection_ref, 'rgb(2018.10.14)',
#                                   'D:/Проекты/Бронницы/Растровые данные')
#
#     shot.to_add_texture_data(texture_shot_address,
#                            directions=directions, window_width=window, distance=distances, grad_count=gradations,
#                            texture_data_linked_to_spec=True,
#                            to_save_clipped_image_like_high_res=False)
#     shot.to_save_data_shot(shot_name, intermediate_data_directory)

# # Проект Бронницы (WorldView)
# import numpy as np
# from numpy import pi
#
# import DataShot
#
# if __name__ == "__main__":
#    spectral_shot_address = 'D:/Проекты/Бронницы/Растровые данные/058041098010_01/058041098010_01_P001_MUL/' \
#                            '11JUL28090720-M2AS-058041098010_01_P001.XML'
#    texture_shot_address = 'D:/Проекты/Бронницы/Растровые данные/058041098010_01/058041098010_01_P001_PAN/' \
#                           '11JUL28090720-P2AS-058041098010_01_P001.TIF'
#    border_shape_address = 'D:/Проекты/Бронницы/Векторные данные/Border.shp'
#    intermediate_data_directory = 'D:/Проекты/Бронницы/Промежуточные данные'
#
#    window = 60
#    distances = [1]
#    gradations = [100]
#    directions = [0]
#
#    texture_list = None
#
#    shot_name = 'Shot_WorldView_win_60_grad_100'
#    shot = DataShot.DataShot(shot_name, border_shape_address, False)
#    shot.to_add_spectral_data_from_worldview2(['BAND_B', 'BAND_G', 'BAND_Y', 'BAND_R', 'BAND_RE', 'BAND_N', 'BAND_N2'],
#                                              ['blue', 'green', 'yellow', 'red', 'red_edge', 'nir', 'nir2'],
#                                              spectral_shot_address)
#    shot.to_make_rgb('red', 'green', 'blue')
#    shot.to_save_image_as_geotiff(shot.rgb_image, shot.spec_geo_trans, shot.spec_projection_ref, 'rgb', intermediate_data_directory)
#    shot.to_add_texture_data(texture_shot_address,
#                             directions=directions, window_width=window, distance=distances, grad_count=gradations,
#                             texture_data_linked_to_spec=True,
#                             to_save_clipped_image_like_high_res=False)
#    shot.to_save_data_shot(shot_name, intermediate_data_directory)

# # Проект Бронницы (Sentinel)
# import numpy as np
# from numpy import pi
#
# import DataShot
#
# if __name__ == "__main__":
#     path = 'D:/Проекты/Бронницы/Растровые данные/Sentinel 2 (2017.11.08)/GRANULE/L1C_T37UDB_A003522_20171108T084135/IMG_DATA/T37UDB_20171108T084139'
#     tiff_name = 'Sentinel 2 (2017.11.08)'
#     spectral_shot_addresses = [path + '_B01.jp2',
#                                path + '_B02.jp2',
#                                path + '_B03.jp2',
#                                path + '_B04.jp2',
#                                path + '_B05.jp2',
#                                path + '_B06.jp2',
#                                path + '_B07.jp2',
#                                path + '_B08.jp2',
#                                path + '_B8A.jp2',
#                                path + '_B09.jp2',
#                                path + '_B10.jp2',
#                                path + '_B11.jp2',
#                                path + '_B12.jp2'
#                                ]
#     texture_shot_address = 'D:/Проекты/Бронницы/Растровые данные/058041098010_01/058041098010_01_P001_PAN/' \
#                            '11JUL28090720-P2AS-058041098010_01_P001.TIF'
#     border_shape_address = 'D:/Проекты/Бронницы/Векторные данные/Border.shp'
#     intermediate_data_directory = 'D:/Проекты/Бронницы/Промежуточные данные'
#
#     window = 80
#     distances = [1]
#     gradations = [50]
#     directions = [0]
#
#     texture_list = None
#
#     shot_name = 'Shot_Sentinel_win_80_grad_50_full_(2016.07.24)'
#
#     upper_wavelengths = [0.523, 0.578, 0.680, 0.903, 1.655, 2.280, 1, 1, 1, 1, 1, 1, 1]
#     lower_wavelengths = [0.458, 0.543, 0.650, 0.788, 1.565, 2.100, 1, 1, 1, 1, 1, 1, 1]
#
#     shot = DataShot.DataShot(shot_name, border_shape_address, False)
#     shot.to_add_spectral_data_from_images(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'],
#                                         ['coastal aerosol', 'blue', 'green', 'red', 'VRE1', 'VRE2', 'VRE3', 'nir', 'VRE4', 'water vapour', 'swir0', 'swir1', 'swir2'],
#                                         spectral_shot_addresses,
#                                         [None, None, None, None, None, None, None, None, None, None, None, None, None], upper_wavelengths, lower_wavelengths,
#                                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#
#     shot.to_make_rgb('red', 'green', 'blue')
#     hypercube = shot.to_combine_data_in_hypercube(['coastal aerosol', 'blue', 'green', 'red', 'VRE1', 'VRE2', 'VRE3', 'nir', 'VRE4', 'water vapour', 'swir0', 'swir1', 'swir2'], sort=False)
#     hypercube = np.swapaxes(np.swapaxes(hypercube, 0, 2), 0, 1)
#     shot.to_save_image_as_geotiff(hypercube, shot.spec_geo_trans, shot.spec_projection_ref, tiff_name, intermediate_data_directory)
#     #shot.to_save_image_as_geotiff(shot.rgb_image, shot.spec_geo_trans, shot.spec_projection_ref, 'Sentinel rgb (2021.07.20)', intermediate_data_directory)
#
#     # shot.to_add_texture_data(texture_shot_address,
#     #                        directions=directions, window_width=window, distance=distances, grad_count=gradations,
#     #                        texture_data_linked_to_spec=True,
#     #                        to_save_clipped_image_like_high_res=False)
#     # shot.to_save_data_shot(shot_name, intermediate_data_directory)

# # Проект Бронницы (Sentinel corr)
# import numpy as np
# from numpy import pi
#
# import DataShot
# import tifffile as tiff
# from matplotlib import pyplot as plt
#
# if __name__ == "__main__":
#     #A = tiff.imread('C:/Users/Sergey/Downloads/LC08_L2SP_172029_20210829_20210901_02_T1_SR_B4.TIF')
#     #plt.imshow(A)
#
#     path = 'D:/Проекты/Велики Столак/Raster/qgis_DOS_Sentinel 2021.05.04/RT_T34TCP_20210504T094031'
#     spectral_shot_addresses = [
#                                path + '_B01.tif',#.jp2',
#                                path + '_B02.tif',#.jp2',
#                                path + '_B03.tif',#.jp2',
#                                path + '_B04.tif',#.jp2',
#                                path + '_B08.tif',#.jp2',
#                                path + '_B11.tif',#.jp2',
#                                path + '_B12.tif'#.jp2'
#                                ]
#     texture_shot_address = 'D:/Проекты/Бронницы/Растровые данные/058041098010_01/058041098010_01_P001_PAN/' \
#                            '11JUL28090720-P2AS-058041098010_01_P001.TIF'
#     border_shape_address = 'D:/Проекты/Велики Столак/Vector/Border_2.shp'
#     intermediate_data_directory = 'D:/Проекты/Велики Столак/inter_data'
#
#     window = 80
#     distances = [1]
#     gradations = [50]
#     directions = [0]
#
#     texture_list = None
#
#     shot_name = 'qgis_DOS_Shot_Sentinel_win_80_grad_50_full_aerosol_(2021.05.04)'
#
#     upper_wavelengths = [0.443, 0.523, 0.578, 0.680, 0.903, 1.655, 2.280]
#     lower_wavelengths = [0.443, 0.458, 0.543, 0.650, 0.788, 1.565, 2.100]
#
#     shot = DataShot.DataShot(shot_name, border_shape_address, False)
#     shot.to_add_spectral_data_from_images(['B1', 'B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                                         ['aerosol', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                         spectral_shot_addresses,
#                                         [None, None, None, None, None, None, None], upper_wavelengths, lower_wavelengths,
#                                         [1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0])
#
#     #shot.to_add_texture_data(texture_shot_address,
#     #                       directions=directions, window_width=window, distance=distances, grad_count=gradations,
#     #                       texture_data_linked_to_spec=True,
#     #                       to_save_clipped_image_like_high_res=False)
#     shot.to_save_data_shot(shot_name, intermediate_data_directory)

# Проект Сербия
import numpy as np
from numpy import pi

import DataShot
import tifffile as tiff
from matplotlib import pyplot as plt

if __name__ == "__main__":
    date = '2020.09.11'
    path = 'D:\Проекты\Велики Столак\Raster\Level-2A\Clipped Sentinel ' + date
    spectral_shot_addresses = [
                               path + '/Sentinel_B1_B9_B10_60m_' + date + '.tif',
                               path + '/Sentinel_B2_B3_B4_B8_10m_' + date + '.tif',
                               path + '/Sentinel_B2_B3_B4_B8_10m_' + date + '.tif',
                               path + '/Sentinel_B2_B3_B4_B8_10m_' + date + '.tif',
                               path + '/Sentinel_B5_B6_B7_B8A_B11_B12_20m_' + date + '.tif',
                               path + '/Sentinel_B5_B6_B7_B8A_B11_B12_20m_' + date + '.tif',
                               path + '/Sentinel_B5_B6_B7_B8A_B11_B12_20m_' + date + '.tif',
                               path + '/Sentinel_B2_B3_B4_B8_10m_' + date + '.tif',
                               path + '/Sentinel_B5_B6_B7_B8A_B11_B12_20m_' + date + '.tif',
                               path + '/Sentinel_B1_B9_B10_60m_' + date + '.tif',
                               path + '/Sentinel_B5_B6_B7_B8A_B11_B12_20m_' + date + '.tif',
                               path + '/Sentinel_B5_B6_B7_B8A_B11_B12_20m_' + date + '.tif'
                               ]
    texture_shot_address = 'D:/Проекты/Бронницы/Растровые данные/058041098010_01/058041098010_01_P001_PAN/' \
                           '11JUL28090720-P2AS-058041098010_01_P001.TIF'
    border_shape_address = 'D:/Проекты/Велики Столак/Vector/Border_2.shp'
    intermediate_data_directory = 'D:/Проекты/Велики Столак/inter_data'

    window = 80
    distances = [1]
    gradations = [50]
    directions = [0]

    texture_list = None

    shot_name = 'Shot_Veliki_Stolek_' + date

    upper_wavelengths = [0.454, 0.523, 0.578, 0.680, 0.711, 0.747, 0.793, 0.903, 0.875, 0.953, 1.655, 2.280]
    lower_wavelengths = [0.433, 0.458, 0.543, 0.650, 0.696, 0.732, 0.773, 0.788, 0.854, 0.933, 1.565, 2.100]

    shot = DataShot.DataShot(shot_name, border_shape_address, False)
    shot.to_add_spectral_data_from_images(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'],
                                        ['aerosol', 'blue', 'green', 'red', 'red_edge1', 'red_edge2', 'red_edge3', 'nir', 'narrow_nir', 'water_vapour', 'swir1', 'swir2'],
                                        spectral_shot_addresses,
                                        [0, 0, 1, 2, 0, 1, 2, 3, 3, 1, 4, 5], upper_wavelengths, lower_wavelengths,
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    #shot.to_add_texture_data(texture_shot_address,
    #                       directions=directions, window_width=window, distance=distances, grad_count=gradations,
    #                       texture_data_linked_to_spec=True,
    #                       to_save_clipped_image_like_high_res=False)
    shot.to_save_data_shot(shot_name, intermediate_data_directory)

# # Проект Бронницы (Landsat)
# import numpy as np
# from numpy import pi
#
# import DataShot
#
# if __name__ == "__main__":
#     spectral_shot_address = 'D:/Проекты/Бронницы/Растровые данные/Landsat 8 (2014.07.28)/LC08_L2SP_177021_20140728_20200911_02_T1_MTL.txt'
#     texture_shot_address = 'D:/Проекты/Бронницы/Растровые данные/058041098010_01/058041098010_01_P001_PAN/' \
#                            '11JUL28090720-P2AS-058041098010_01_P001.TIF'
#     border_shape_address = 'D:/Проекты/Бронницы/Векторные данные/Border.shp'
#     intermediate_data_directory = 'D:/Проекты/Бронницы/Промежуточные данные'
#
#     window = 80
#     distances = [1]
#     gradations = [50]
#     directions = [0]
#
#     texture_list = None
#
#     shot_name = 'Shot_Landsat_win_80_grad_50'
#
#     upper_wavelengths = [0.523, 0.578, 0.680, 0.903, 1.655, 2.280]
#     lower_wavelengths = [0.458, 0.543, 0.650, 0.788, 1.565, 2.100]
#
#     shot = DataShot.DataShot(shot_name, border_shape_address, False)
#     shot.to_add_spectral_data_from_landsat8(['BAND_2', 'BAND_3', 'BAND_4', 'BAND_5', 'BAND_6', 'BAND_7'],
#                                           ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                         spectral_shot_address)
#     shot.to_add_texture_data(texture_shot_address,
#                            directions=directions, window_width=window, distance=distances, grad_count=gradations,
#                            texture_data_linked_to_spec=True,
#                            to_save_clipped_image_like_high_res=False)
#     shot.to_save_data_shot(shot_name, intermediate_data_directory)

# # Проект Бронницы (Resurs-P)
# import numpy as np
# from numpy import pi
#
# import DataShot
#
# if __name__ == "__main__":
#    spectral_shot_address = 4 * ['D:/Проекты/Бронницы/Растровые данные/Ресурс-П (14.05.2018)/Resurs-P(32637).tif']
#    texture_shot_address = 'D:/Проекты/Бронницы/Растровые данные/058041098010_01/058041098010_01_P001_PAN/' \
#                           '11JUL28090720-P2AS-058041098010_01_P001.TIF'
#    border_shape_address = 'D:/Проекты/Бронницы/Векторные данные/Border_resurs.shp'
#    intermediate_data_directory = 'D:/Проекты/Бронницы/Промежуточные данные'
#
#    upper_wavelengths = [0.51, 0.58, 0.7, 0.9]
#    lower_wavelengths = [0.43, 0.51, 0.6, 0.7]
#
#    window = 80
#    distances = [1]
#    gradations = [50]
#    directions = [0]
#
#    texture_list = None
#
#    shot_name = 'Shot_Resurs_win_80_grad_50'
#    shot = DataShot.DataShot(shot_name, border_shape_address, False)
#    shot.to_add_spectral_data_from_images(['B1', 'B2', 'B3', 'B4'],
#                                         ['blue', 'green', 'red', 'nir'],
#                                         spectral_shot_address,
#                                         [0, 1, 2, 3], upper_wavelengths, lower_wavelengths,
#                                         [1, 1, 1, 1], [0, 0, 0, 0])
#
#    shot.to_add_texture_data(texture_shot_address,
#                            directions=directions, window_width=window, distance=distances, grad_count=gradations,
#                            texture_data_linked_to_spec=True,
#                            to_save_clipped_image_like_high_res=False)
#    shot.to_save_data_shot(shot_name, intermediate_data_directory)

# # Создание шотов для задачи определения полноты для участка Валуйского лесничества
#
# import numpy as np
# from numpy import  pi
#
# import DataShot
#
# if __name__ == "__main__":
#   texture_shot_addresses = 'D:/Проекты/Структурные индексы/Растровые данные/17AUG03084409-P2AS-058041098030_01_P001.TIF'
#   #texture_shot_addresses = 'D:/Data/Спектральные изображения/Тверская область (север)/WorldView 2/056009302010_01_P002_MUL/' \
#   #                         '16JUN25085338-M2AS-056009302010-01-P002.TIF'
#   border_shape_address = 'D:/Проекты/Структурные индексы/Векторные данные/Граница района.shp'
#   intermediate_data_directory = 'D:/Проекты/Структурные индексы/Промежуточные данные/'
#
#   window = 64
#   distances = [1]
#   gradations = [100]
#   directions = [0]
#
#   texture_list = ['SDGL', 'Contrast', 'Entropy']
#
#   shot_name = 'p_WorldView_shot_100'
#   shot = DataShot.DataShot(shot_name, border_shape_address, False)
#   shot.to_add_texture_data(texture_shot_addresses, texture_names=texture_list,
#                            directions=directions, window_width=window, distance=distances, grad_count=gradations,
#                            to_save_clipped_image_like_high_res=False)
#   shot.to_save_data_shot(shot_name, intermediate_data_directory)

# # Создание шотов для задачи определения полноты для участка Валуйского лесничества
#
# import numpy as np
# from numpy import pi
#
# import DataShot
#
# if __name__ == "__main__":
#    texture_shot_addresses = 'D:/Проекты/Структурные индексы/Растровые данные/17AUG03084409-M2AS-058041098030_01_P001.TIF'
#    border_shape_address = 'D:/Проекты/Структурные индексы/Векторные данные/Граница района.shp'
#    intermediate_data_directory = 'D:/Проекты/Структурные индексы/Промежуточные данные/'
#
#    window = 16
#    distances = [1]
#    gradations = [100]
#    directions = [0]
#
#    texture_list = ['SDGL', 'Contrast', 'Entropy']
#
#    shot_name = 'WorldView_shot_100'
#    shot = DataShot.DataShot(shot_name, border_shape_address, False)
#    shot.to_add_texture_data(texture_shot_addresses, texture_names=texture_list,
#                             directions=directions, window_width=window, distance=distances, grad_count=gradations,
#                             to_save_clipped_image_like_high_res=False)
#    shot.to_save_data_shot(shot_name, intermediate_data_directory)

# import numpy as np
# from numpy import pi
#
# import DataShot

# if __name__ == "__main__":
#     spectral_shot_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/' \
#                                  'Саватьевское лес-во (WorldView 2)/Общие данные/Rastrs/Спектр/WorldView 2/' \
#                                  '056009302010_01_P002_MUL/16JUN25085338-M2AS-056009302010-01-P002.XML'
#     spectral_shot_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/' \
#                                  'Саватьевское лес-во (WorldView 2)/Общие данные/Rastrs/Спектр/WorldView 2/' \
#                                  '056009302010_01_P001_MUL/16JUN25085327-M2AS-056009302010-01-P001.XML'
#     texture_shot_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/' \
#                                 'Саватьевское лес-во (WorldView 2)/Общие данные/Rastrs/Текстурные данные/Запад/' \
#                                 '056009302010_01_P002_PAN/16JUN25085338-P2AS-056009302010_01_P002.TIF'
#     texture_shot_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/' \
#                                 'Саватьевское лес-во (WorldView 2)/Общие данные/Rastrs/Текстурные данные/Восток/' \
#                                 '056009302010_01_P001_PAN/16JUN25085327-P2AS-056009302010_01_P001.TIF'
#     border_shape_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/' \
#                                 'Саватьевское лес-во (WorldView 2)/Общие данные/Shapes/Polygon border (west).shp'
#     border_shape_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/' \
#                                 'Саватьевское лес-во (WorldView 2)/Общие данные/Shapes/Polygon border (east).shp'
#     intermediate_data_directory = 'D:/Проекты/Классификация (спектальные и текстурные данные)/' \
#                                   'Саватьевское лес-во (WorldView 2)/Общие данные/Data shots'
#
#     windows = [80]
#     distances = [[1]]
#     gradations = [[100]]
#     directions=[0, 3 * pi / 4]
#
#     texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#
#     for win in windows:
#         for dist in distances:
#             for grad in gradations:
                #west_shot_name = 'west_win_' + str(win) + '_dist_' + str(dist) + '_grad_' + str(grad[0])
                #shot = DataShot.DataShot(west_shot_name, border_shape_address_west, False)
                #shot.to_add_spectral_data_from_worldview2(['BAND_B', 'BAND_G', 'BAND_Y', 'BAND_R', 'BAND_RE', 'BAND_N', 'BAND_N2'],
                #                                         ['blue', 'green', 'yellow', 'red', 'red_edge', 'nir', 'nir2'],
                #                                         spectral_shot_address_west)
                #shot.to_add_texture_data(texture_shot_address_west, texture_names=texture_list,
                #                         directions=directions, window_width=win, distance=dist, grad_count=grad,
                #                         texture_data_linked_to_spec=True,
                #                         to_save_clipped_image_like_high_res=False)
                #shot.to_save_data_shot(west_shot_name, intermediate_data_directory)

                # east_shot_name = 'east_win_' + str(win) + '_dist_' + str(dist) + '_grad_' + str(grad[0])
                # shot = DataShot.DataShot(east_shot_name, border_shape_address_east, False)
                # shot.to_add_spectral_data_from_worldview2(['BAND_B', 'BAND_G', 'BAND_Y', 'BAND_R', 'BAND_RE', 'BAND_N', 'BAND_N2'],
                #                                          ['blue', 'green', 'yellow', 'red', 'red_edge', 'nir', 'nir2'],
                #                                          spectral_shot_address_east)
                # shot.to_add_texture_data(texture_shot_address_east, texture_names=texture_list,
                #                          directions=directions, window_width=win, distance=dist, grad_count=grad,
                #                          texture_data_linked_to_spec=True,
                #                          to_save_clipped_image_like_high_res=False)
                # shot.to_save_data_shot(east_shot_name, intermediate_data_directory)


#import numpy as np
#from numpy import pi
#
#import DataShot
#
#if __name__ == "__main__":
#   spectral_shot_addresses = ['D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/'
#                              'Общие данные/Rastrs/Спектр/Sentinel 2/GRANULE/IMG_DATA/'
#                              'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37VCD_B02.jp2',
#                              'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/'
#                              'Общие данные/Rastrs/Спектр/Sentinel 2/GRANULE/IMG_DATA/'
#                              'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37VCD_B03.jp2',
#                              'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/'
#                              'Общие данные/Rastrs/Спектр/Sentinel 2/GRANULE/IMG_DATA/'
#                              'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37VCD_B04.jp2',
#                              'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/'
#                              'Общие данные/Rastrs/Спектр/Sentinel 2/GRANULE/IMG_DATA/'
#                              'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37VCD_B08.jp2',
#                              'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/'
#                              'Общие данные/Rastrs/Спектр/Sentinel 2/GRANULE/IMG_DATA/'
#                              'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37VCD_B11.jp2',
#                              'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/'
#                              'Общие данные/Rastrs/Спектр/Sentinel 2/GRANULE/IMG_DATA/'
#                              'S2A_OPER_MSI_L1C_TL_SGS__20160724T123647_A005681_T37VCD_B12.jp2',
#                              ]
#   texture_shot_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                               'Общие данные/Rastrs/Текстурные данные/Запад/056009302010_01_P002_PAN/' \
#                               '16JUN25085338-P2AS-056009302010_01_P002.TIF'
#   texture_shot_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                               'Общие данные/Rastrs/Текстурные данные/Восток/056009302010_01_P001_PAN/' \
#                               '16JUN25085327-P2AS-056009302010_01_P001.TIF'
#   border_shape_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                               'Общие данные/Shapes/Polygon border (test).shp'
#   border_shape_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                               'Общие данные/Shapes/Polygon border (east).shp'
#   intermediate_data_directory = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во2/' \
#                                 'Общие данные/Data shots'
#
#   windows = [80]
#   distances = [[1]]
#   gradations = [[100]]
#   directions=[0, 3 * pi / 4]
#
#   texture_list = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#
#   upper_wavelengths = [0.523, 0.578, 0.680, 0.903, 1.655, 2.280]
#   lower_wavelengths = [0.458, 0.543, 0.650, 0.788, 1.565, 2.100]
#
#
#   for win in windows:
#       for dist in distances:
#           for grad in gradations:
#               west_shot_name = 'test_win_' + str(win) + '_dist_' + str(dist) + '_grad_' + str(grad[0])
#               shot = DataShot.DataShot(west_shot_name, border_shape_address_west, False)
#               shot.to_add_spectral_data_from_images(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#                                                     ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                                     spectral_shot_addresses,
#                                                     [None, None, None, None, None, None], upper_wavelengths, lower_wavelengths,
#                                                     [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0])
#               shot.to_add_texture_data(texture_shot_address_west, texture_names=texture_list,
#                                        directions=directions, window_width=win, distance=dist, grad_count=grad,
#                                        texture_data_linked_to_spec=True,
#                                        to_save_clipped_image_like_high_res=False)
#               shot.to_save_data_shot(west_shot_name, intermediate_data_directory)
#
#               # east_shot_name = 'sel_east_win_' + str(win) + '_dist_' + str(dist) + '_grad_' + str(grad[0])
#               # shot = DataShot.DataShot(east_shot_name, border_shape_address_east, False)
#               # shot.to_add_spectral_data_from_images(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
#               #                                       ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#               #                                       spectral_shot_addresses,
#               #                                       [0, 0, 0, 0, 0, 0], upper_wavelengths, lower_wavelengths,
#               #                                       [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0])
#               # shot.to_add_texture_data(texture_shot_address_east,
#               #                          directions=directions, window_width=win, distance=dist, grad_count=grad,
#               #                          texture_data_linked_to_spec=True,
#               #                          to_save_clipped_image_like_high_res=False)
#               # shot.to_save_data_shot(east_shot_name, intermediate_data_directory)
##
#    #spectral_shot_address = 'D:/Data/Спектральные изображения/Тверская область (север)/Sentinel 2/2019.06.04/' \
#    #                        'S2B_MSIL1C_20190604T084609_N0207_R107_T37VCD_20190604T113755.SAFE/MTD_MSIL1C.xml'
#    ##texture_shot_address = 'D:/Проекты/Текстуры/Растр/high res 100 (left).tif'
    #texture_shot_address_left = 'D:/Проекты/Текстуры/Растр/high res 100 (left).tif'
    #texture_shot_address_right = 'D:/Проекты/Текстуры/Растр/high res 100 (right).tif'
    #border_shape_address_left = 'D:/Проекты/Текстуры/Shape/Polygon border (left).shp'
    #border_shape_address_right = 'D:/Проекты/Текстуры/Shape/Polygon border (right).shp'
    #rast_files_directory = 'D:/Проекты/Текстуры/Растр'
    #intermediate_data_directory = 'D:/Проекты/Текстуры/Промежуточные результаты'
    #
    #shot = DataShot.DataShot('test_data_shot', border_shape_address_left, False)
    #shot.to_add_spectral_data_from_sentinel2(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
    #                                         ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
    #                                         spectral_shot_address)
    #shot.to_add_texture_data(texture_shot_address_left,
    #                          directions=[0, pi / 4, pi / 2, 3 * pi / 4], window_width=30, texture_data_linked_to_spec=True,
    #                          to_save_clipped_image_like_high_res=False, grad_count=None)
    #
    #shot.to_save_data_shot('left_win_30_dist_1', intermediate_data_directory)
    #
    #shot = DataShot.DataShot('test_data_shot', border_shape_address_right, False)
    #shot.to_add_spectral_data_from_sentinel2(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
    #                                         ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
    #                                         spectral_shot_address)
    #shot.to_add_texture_data(texture_shot_address_right,
    #                         directions=[0, pi / 4, pi / 2, 3 * pi / 4], window_width=30,
    #                         texture_data_linked_to_spec=True,
    #                         to_save_clipped_image_like_high_res=False, grad_count=None)
    #
    #shot.to_save_data_shot('right_win_30_dist_1', intermediate_data_directory)
    #
    #shot = DataShot.DataShot('test_data_shot', border_shape_address_left, False)
    #shot.to_add_spectral_data_from_sentinel2(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
    #                                         ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
    #                                         spectral_shot_address)
    #shot.to_add_texture_data(texture_shot_address_left,
    #                         directions=[0, pi / 4, pi / 2, 3 * pi / 4], window_width=40,
    #                         texture_data_linked_to_spec=True,
    #                         to_save_clipped_image_like_high_res=False, grad_count=None)
    #
    #shot.to_save_data_shot('left_win_40_dist_1', intermediate_data_directory)
    #
    #shot = DataShot.DataShot('test_data_shot', border_shape_address_right, False)
    #shot.to_add_spectral_data_from_sentinel2(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
    #                                         ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
    #                                         spectral_shot_address)
    #shot.to_add_texture_data(texture_shot_address_right,
    #                         directions=[0, pi / 4, pi / 2, 3 * pi / 4], window_width=40,
    #                         texture_data_linked_to_spec=True,
    #                         to_save_clipped_image_like_high_res=False, grad_count=None)
    #
    #shot.to_save_data_shot('right_win_40_dist_1', intermediate_data_directory)
    #
    #shot = DataShot.DataShot('test_data_shot', border_shape_address_left, False)
    #shot.to_add_spectral_data_from_sentinel2(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
    #                                         ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
    #                                         spectral_shot_address)
    #shot.to_add_texture_data(texture_shot_address_left,
    #                         directions=[0, pi / 4, pi / 2, 3 * pi / 4], window_width=50,
    #                         texture_data_linked_to_spec=True,
    #                         to_save_clipped_image_like_high_res=False, grad_count=None)
    #
    #shot.to_save_data_shot('left_win_50_dist_1', intermediate_data_directory)
    #
    #shot = DataShot.DataShot('test_data_shot', border_shape_address_right, False)
    #shot.to_add_spectral_data_from_sentinel2(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
    #                                         ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
    #                                         spectral_shot_address)
    #shot.to_add_texture_data(texture_shot_address_right,
    #                         directions=[0, pi / 4, pi / 2, 3 * pi / 4], window_width=50,
    #                         texture_data_linked_to_spec=True,
    #                         to_save_clipped_image_like_high_res=False, grad_count=None)
    #
    #shot.to_save_data_shot('right_win_50_dist_1', intermediate_data_directory)
    # shot.to_save_data_shot('shot_right_100_win_10_dist_3', intermediate_data_directory)

    # shot.to_make_rgb('red', 'red', 'red',
    #                  red_band_add=0, green_band_add=0, blue_band_add=0,
    #                  red_band_mult=1, green_band_mult=1, blue_band_mult=1)
    # shot.to_save_image_as_geotiff(shot.rgb_image, shot.spec_geo_trans, shot.spec_projection_ref, '2',
    #                               'D:/Data')
    # shot.to_save_image_as_geotiff(shot.high_res_image, shot.texture_geo_trans, shot.texture_projection_ref, '1',
    #                               'D:/Data')


    #windows = [80]
    #for win in windows:
    #    for dist in [1]:
    #        shot = DataShot.DataShot('test_data_shot', border_shape_address, False)
    #        shot.to_add_texture_data(texture_shot_address,
    #                                 directions=[0, pi / 4, pi / 2, 3 * pi / 4], window_width=win, distance=dist,
    #                                 grad_count=[60],
    #                                 texture_data_linked_to_spec=False,
    #                                 to_save_clipped_image_like_high_res=False)
    #        hypercube = shot.to_combine_data_in_hypercube()
    #        name = 'shot_right_win_' + str(win) + '_dist_' + str(dist) + '_grad_60'
    #        shot.to_save_data_shot(name, intermediate_data_directory)
    #        print(name)

# import numpy as np
# from numpy import pi
#
# import DataShot
#
# if __name__ == "__main__":
#     spectral_shot_address = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                             'Общие данные/Rastrs/Спектр/Landsat 8/2016.06.29/LC08_L1TP_179020_20160629_20170323_01_T1_MTL.txt'
#     texture_shot_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                                 'Общие данные/Rastrs/Текстурные данные/Запад/056009302010_01_P002_PAN/' \
#                                 '16JUN25085338-P2AS-056009302010_01_P002.TIF'
#     texture_shot_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                                 'Общие данные/Rastrs/Текстурные данные/Восток/056009302010_01_P001_PAN/' \
#                                 '16JUN25085327-P2AS-056009302010_01_P001.TIF'
#     border_shape_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                                 'Общие данные/Shapes/Polygon border (west).shp'
#     border_shape_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                                 'Общие данные/Shapes/Polygon border (east).shp'
#     intermediate_data_directory = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
#                                   'Общие данные/Data shots'
#
#     windows = [80]
#     distances = [[1]]
#     gradations = [[100]]
#     directions=[0, 3 * pi / 4]
#     texture_names = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#
#     upper_wavelengths = [0.523, 0.578, 0.680, 0.903, 1.655, 2.280]
#     lower_wavelengths = [0.458, 0.543, 0.650, 0.788, 1.565, 2.100]
#
#
#     for win in windows:
#         for dist in distances:
#             for grad in gradations:
#                 # west_shot_name = 'west_win_' + str(win) + '_dist_' + str(dist) + '_grad_' + str(grad[0])
#                 # shot = DataShot.DataShot(west_shot_name, border_shape_address_west, False)
#                 # shot.to_add_spectral_data_from_landsat8(['BAND_2', 'BAND_3', 'BAND_4', 'BAND_5', 'BAND_6', 'BAND_7'],
#                 #                                         ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                 #                                         spectral_shot_address)
#                 # shot.to_add_texture_data(texture_shot_address_west, texture_names=texture_names,
#                 #                          directions=directions, window_width=win, distance=dist, grad_count=grad,
#                 #                          texture_data_linked_to_spec=True,
#                 #                          to_save_clipped_image_like_high_res=False)
#                 # shot.to_save_data_shot(west_shot_name, intermediate_data_directory)
#
#                 east_shot_name = 'east_win_' + str(win) + '_dist_' + str(dist) + '_grad_' + str(grad[0])
#                 shot = DataShot.DataShot(east_shot_name, border_shape_address_east, False)
#                 shot.to_add_spectral_data_from_landsat8(['BAND_2', 'BAND_3', 'BAND_4', 'BAND_5', 'BAND_6', 'BAND_7'],
#                                                         ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
#                                                         spectral_shot_address)
#                 shot.to_add_texture_data(texture_shot_address_east,
#                                          directions=directions, window_width=win, distance=dist, grad_count=grad,
#                                          texture_data_linked_to_spec=True,
#                                          to_save_clipped_image_like_high_res=False)
#                 shot.to_save_data_shot(east_shot_name, intermediate_data_directory)


# import numpy as np
# from numpy import pi
#
# import DataShot
#
# if __name__ == "__main__":
#    spectral_shot_address = 'D:/Проекты/Классификация (спектальные и текстурные данные)/' \
#                              'Саватьевское лес-во (WorldView 2)/Общие данные/Rastrs/Спектр/WorldView 2/' \
#                              '056009302010_01_P002_MUL/16JUN25085338-M2AS-056009302010-01-P002.XML'
#    texture_shot_address_west = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (WorldView 2)/' \
#                                'Общие данные/Rastrs/Текстурные данные/Запад/056009302010_01_P002_PAN/' \
#                                '16JUN25085338-P2AS-056009302010_01_P002.TIF'
#    win = 80
#    dist = [1]
#    grad = [100]
#    directions=[0, 3 * pi / 4]
#    texture_names = ['Autocorrelation', 'ClusterShade', 'Contrast', 'Correlation']
#    # # channels_quan = 87
#    # # layers = list(range(0, channels_quan))
#    # border_shape_address = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (WorldView 2)/' \
#    #                        'Общие данные/Shapes/Polygon border (west).shp'
#    # shot_name = 'west_map_80'
#    # shot_path = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (WorldView 2)/' \
#    #             'Общие данные/Data shots'
#    #
#    # shot = DataShot.DataShot(shot_name, border_shape_address, False)
#    # shot.to_add_spectral_data_from_worldview2(['BAND_B', 'BAND_G', 'BAND_R', 'BAND_N', 'BAND_N2'],
#    #                                          ['blue', 'green', 'red', 'nir', 'nir2'],
#    #                                          spectral_shot_address)
#    # shot.to_add_texture_data(texture_shot_address_west, texture_names=texture_names,
#    #                          directions=directions, window_width=win, distance=dist, grad_count=grad,
#    #                          texture_data_linked_to_spec=True,
#    #                          to_save_clipped_image_like_high_res=False)
#    # shot.to_save_data_shot(shot_name, shot_path)
#
#    spectral_shot_address = 'D:/Проекты/Классификация (спектальные и текстурные данные)/' \
#                            'Саватьевское лес-во (WorldView 2)/Общие данные/Rastrs/Спектр/WorldView 2/' \
#                            '056009302010_01_P001_MUL/16JUN25085327-M2AS-056009302010-01-P001.XML'
#    texture_shot_address_east = 'D:/Проекты/Классификация (спектальные и текстурные данные)/' \
#                                  'Саватьевское лес-во (WorldView 2)/Общие данные/Rastrs/Текстурные данные/Восток/' \
#                                '056009302010_01_P001_PAN/16JUN25085327-P2AS-056009302010_01_P001.TIF'
#    border_shape_address = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (WorldView 2)/' \
#                           'Общие данные/Shapes/Polygon border (east).shp'
#    shot_name = 'east_map_80'
#    shot_path = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (WorldView 2)/' \
#                'Общие данные/Data shots'
#
#    shot = DataShot.DataShot(shot_name, border_shape_address, False)
#    shot.to_add_spectral_data_from_worldview2(['BAND_B', 'BAND_G', 'BAND_R', 'BAND_N', 'BAND_N2'],
#                                              ['blue', 'green', 'red', 'nir', 'nir2'],
#                                              spectral_shot_address)
#    shot.to_add_texture_data(texture_shot_address_east, texture_names=texture_names,
#                             directions=directions, window_width=win, distance=dist, grad_count=grad,
#                             texture_data_linked_to_spec=True,
#                             to_save_clipped_image_like_high_res=False)
#    shot.to_save_data_shot(shot_name, shot_path)
