import numpy as np
import shapefile
import tifffile as tiff
import scipy.misc
from matplotlib import image
from osgeo import gdal
from GeoImageFunctions import to_clip_shot

if __name__ == "__main__":
    shape_path = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
                       'mixed_shape.shp'
    image_path = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
                 'Классификации/Классификация леса на лиственный и хвойный/Результаты/KNN/Sentinel 2/Спектр + текстура/' \
                 'west_map_win_80_dist_[1]_grad_100.tif'
    res_path = 'D:/Проекты/Классификация (спектальные и текстурные данные)/Саватьевское лес-во (Landsat 8)/' \
               'Классификации/Классификация леса на лиственный и хвойный/Результаты'

    polygons = shapefile.Reader(shape_path)
    shapes = polygons.shapes()
    # texture_image_address = 'D:/Data/Высокое разрешение/Савватьевское лесничество/056009302010_01_P002_PAN/16JUN25085338-P2AS-056009302010_01_P002.TIF'
    # polygons = shapefile.Reader('D:/Проекты/Текстуры/Shape/visual_completeness (right).shp')
    # shapes = polygons.shapes()
    # texture_image_address = 'D:/Data/Высокое разрешение/Савватьевское лесничество/056009302010_01_P001_PAN/16JUN25085327-P2AS-056009302010_01_P001.TIF'
    # Чтение изображения
    digital_im = tiff.imread(image_path)

    # Чтение привязки
    rast = gdal.Open(image_path)
    geo_trans = rast.GetGeoTransform()
    projection_ref = rast.GetProjectionRef()
    for i, shp in enumerate(shapes):
        #q = polygons.records()[i][1]
        #id = polygons.records()[i][0]
        new_image = to_clip_shot(digital_im, shp, geo_trans, projection_ref, rectangle_shape=True)
        #address = 'D:/Data/clip samples/Новые фрагменты/' + 'Q_' + str(q) + '_' + str(id) + '.png'
        new_image = np.repeat(new_image[0], 20, axis=0)
        new_image = np.repeat(new_image, 20, axis=1)
        image.imsave(res_path + '/21.png', new_image, cmap='gray')
