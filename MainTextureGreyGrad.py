from DataCollectionFunctions import to_transform_and_clip_image_grey_grad

# Трансформация снимка части Саватьевского лесничества
if __name__ == "__main__":
    max_grad = [255]
    # texture_shot_address = 'D:/Проекты/Структурные индексы/Растровые данные/17AUG03084409-M2AS-058041098030_01_P001.TIF'
    texture_shot_address = 'D:/Проекты/Бронницы/Растровые данные/058041098010_01/058041098010_01_P001_PAN/' \
                           '11JUL28090720-P2AS-058041098010_01_P001.TIF'
    new_texture_shot_directory = 'D:/Проекты/Бронницы/Растровые данные'
    border_shape_address = 'D:/Проекты/Бронницы/Векторные данные/Border.shp'

    to_transform_and_clip_image_grey_grad(texture_shot_address, new_texture_shot_directory, 'graded_texture_shot_255', max_grad,
                                          border_shape_address, rectangle_shape=True)

#if __name__ == "__main__":
#    texture_shot_address = 'D:/Data/Высокое разрешение/Савватьевское лесничество/056009302010_01_P002_PAN/' \
#                           '16JUN25085338-P2AS-056009302010_01_P002.TIF'
#    new_texture_shot_directory = 'D:/Проекты/Текстуры/Растр'
#    to_transform_image_grey_grad(texture_shot_address, new_texture_shot_directory, 'high res 100 (left)', 100)
#
#    texture_shot_address = 'D:/Data/Высокое разрешение/Савватьевское лесничество/056009302010_01_P001_PAN/' \
#                           '16JUN25085327-P2AS-056009302010_01_P001.TIF'
#    new_texture_shot_directory = 'D:/Проекты/Текстуры/Растр'
#    to_transform_image_grey_grad(texture_shot_address, new_texture_shot_directory, 'high res 100 (right)', 100)
#
#    texture_shot_address = 'D:/Data/Высокое разрешение/Савватьевское лесничество/056009302010_01_P002_PAN/' \
#                           '16JUN25085338-P2AS-056009302010_01_P002.TIF'
#    new_texture_shot_directory = 'D:/Проекты/Текстуры/Растр'
#    to_transform_image_grey_grad(texture_shot_address, new_texture_shot_directory, 'high res 255 (left)', 255)
#
#    texture_shot_address = 'D:/Data/Высокое разрешение/Савватьевское лесничество/056009302010_01_P001_PAN/' \
#                           '16JUN25085327-P2AS-056009302010_01_P001.TIF'
#    new_texture_shot_directory = 'D:/Проекты/Текстуры/Растр'
#    to_transform_image_grey_grad(texture_shot_address, new_texture_shot_directory, 'high res 255 (right)', 255)