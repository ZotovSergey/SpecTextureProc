# Разделение выборки на обучающую и тестовую для задачи определения полноты Валуйского лесничества

import DataSamples

if __name__ == "__main__":
    samples_shape_address = 'D:/Проекты/Структурные индексы/Векторные данные/WGS_region.shp'
    train_samples_address = 'D:/Проекты/Структурные индексы/Векторные данные/train_samples_reg.shp'
    test_samples_address = 'D:/Проекты/Структурные индексы/Векторные данные/test_samples_reg.shp'
    name_of_field_of_separation = 'SKAL1'
    test_size = 0.5
    random_state = 1

    DataSamples.to_select_samples(samples_shape_address, name_of_field_of_separation,
                                  train_samples_address=train_samples_address,
                                  test_samples_address=test_samples_address, test_size=test_size, random_state=1,
                                  stratify=True)
