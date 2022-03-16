import DataShot

if __name__ == "__main__":
    date = '2020.09.11'
    shot_address = 'D:/Проекты/Велики Столак/inter_data/Shot_Veliki_Stolek_' + date + '.file'
    result_directory = 'D:/Проекты/Велики Столак/results'

    shot = DataShot.to_load_data_shot(shot_address)

    shot.to_make_ndvi('red', 'nir')
    shot.to_save_image_as_geotiff(shot.ndvi, shot.spec_geo_trans, shot.spec_projection_ref, 'ndvi_' + date,
                                  result_directory, normed=False, byte=False)
    shot.to_make_nbr('nir', 'swir1')
    shot.to_save_image_as_geotiff(shot.nbr, shot.spec_geo_trans, shot.spec_projection_ref, 'nbr1_' + date,
                                  result_directory, normed=False, byte=False)
    shot.to_make_nbr('nir', 'swir2')
    shot.to_save_image_as_geotiff(shot.nbr, shot.spec_geo_trans, shot.spec_projection_ref, 'nbr2_' + date,
                                  result_directory, normed=False, byte=False)
    shot.to_make_ndre('red_edge2', 'nir')
    shot.to_save_image_as_geotiff(shot.ndre, shot.spec_geo_trans, shot.spec_projection_ref, 'ndre_' + date,
                                  result_directory, normed=False, byte=False)