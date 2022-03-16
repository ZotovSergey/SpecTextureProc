import numpy as np
from skimage.feature import greycomatrix


def to_calc_textures(window, directions, texture_list, dist=1):
    glcm = greycomatrix(np.int32(window), dist, directions, int(np.max(window)) + 1, symmetric=True, normed=True)
    texture_dict = greycoprops_dir(window, glcm, texture_list)
    return texture_dict


def greycoprops_dir(window, glcm, texture_list):
    # матрица glcm по направлениям сопряженности
    textures = []
    for i in range(glcm.shape[2]):
        textures_directions = []
        for j in range(glcm.shape[3]):
            glcm_direction = glcm[:, :, i, j]
            textures_directions.append(greycoprops_mat(window, glcm_direction, texture_list))
        textures.append(textures_directions)
    texture_dict = {}
    for texture_name in texture_list:
        texture_dict.update({texture_name: np.zeros(np.array(textures).shape)})
    for i, textures_direction_dict in enumerate(textures):
        for j, texture in enumerate(textures_direction_dict):
            for texture_name in texture_list:
                texture_dict[texture_name][i][j] = texture[texture_name]
    return texture_dict


def greycoprops_mat(window, glcm, texture_list):
    glcm_width = len(glcm)
    I = np.repeat(np.swapaxes(np.array([np.arange(1, glcm_width + 1)]), 0, 1), glcm_width, axis=1)
    J = np.repeat(np.array([np.arange(1, glcm_width + 1)]), glcm_width, axis=0)
    mui = 0.0
    sigi = 0.0
    muj = 0.0
    sigj = 0.0
    hxy = 0.0
    normed_glcm_x = 0
    normed_glcm_y = 0
    texture_dict = {}
    glcm_sum = np.sum(glcm)
    if glcm_sum != 0:
        normed_glcm = glcm / glcm_sum
    else:
        normed_glcm = glcm
    is_sum_average = False
    is_hxy = False
    is_mui = False
    if any(['ClusterProminence' in texture_list, 'Correlation' in texture_list, 'ClusterShade' in texture_list]):
        mui = mean_index(I, normed_glcm)
        is_mui = True
        sigi = std_index(I, normed_glcm, mui)
        muj = mean_index(J, normed_glcm)
        sigj = std_index(J, normed_glcm, muj)
    if any(['InfMeasureCorr1' in texture_list, 'InfMeasureCorr2' in texture_list]):
        normed_glcm_x = np.sum(normed_glcm, axis=1)
        normed_glcm_y = np.sum(normed_glcm, axis=0)

        x_temp = np.log(normed_glcm)
        x_temp[np.isinf(x_temp)] = 0
        hxy = -np.sum(normed_glcm * x_temp)
        is_hxy = True
    # Вычисление Autocorrelation
    if 'Autocorrelation' in texture_list:
        texture_dict.update({'Autocorrelation': np.sum(I * J ** 2 * normed_glcm)})
    # Вычисление ClusterProminence
    if 'ClusterProminence' in texture_list:
        texture_dict.update({'ClusterProminence': np.sum((I + J - mui - muj) ** 4 * normed_glcm)})
    # Вычисление ClusterShade
    if 'ClusterShade' in texture_list:
        texture_dict.update({'ClusterShade': np.sum((I + J - mui - muj) ** 3 * normed_glcm)})
    # Вычисление Contrast
    if 'Contrast' in texture_list:
        texture_dict.update({'Contrast': np.sum((I - J) ** 2 * normed_glcm)})
    # Вычисление Correlation
    if 'Correlation' in texture_list:
        texture_dict.update({'Correlation': np.sum(normed_glcm * (I - mui) * (J - muj) / (sigi * sigj))})
    # Вычисление DiffEntropy
    if 'DiffEntropy' in texture_list:
        x_temp1 = p_X_minus_Y(normed_glcm)
        x_temp2 = np.log(x_temp1)
        x_temp2[np.isinf(x_temp2)] = 0
        texture_dict.update({'DiffEntropy': -np.sum(x_temp1 * x_temp2)})
    # Вычисление DiffVariance
    if 'DiffVariance' in texture_list:
        udiff = np.sum(np.arange(0, glcm_width) * p_X_minus_Y(normed_glcm))
        texture_dict.update(
            {'DiffVariance': np.sum(((np.arange(0, glcm_width) - udiff) ** 2) * p_X_minus_Y(normed_glcm))})
    # Вычисление Dissimilarity
    if 'Dissimilarity' in texture_list:
        texture_dict.update({'Dissimilarity': np.sum(np.abs(I - J) * normed_glcm)})
    # Вычисление Energy
    if 'Energy' in texture_list:
        texture_dict.update({'Energy': np.sum(normed_glcm ** 2)})
    # Вычисление Entropy
    if 'Entropy' in texture_list:
        if is_hxy:
            texture_dict.update({'Entropy': hxy})
        else:
            x_temp = np.log(normed_glcm)
            x_temp[np.isinf(x_temp)] = 0
            texture_dict.update({'Entropy': -np.sum(normed_glcm * x_temp)})
    # Вычисление Homogeneity
    if 'Homogeneity' in texture_list:
        texture_dict.update({'Homogeneity': np.sum(normed_glcm / (abs(I - J) + 1))})
    # Вычисление Homogeneity2
    if 'Homogeneity2' in texture_list:
        texture_dict.update({'Homogeneity2': np.sum(normed_glcm / ((I - J) ** 2 + 1))})
    # Вычисление InfMeasureCorr1
    if 'InfMeasureCorr1' in texture_list:
        x_temp = np.log(np.dot(np.array([normed_glcm_x]).T, np.array([normed_glcm_y])))
        x_temp[np.isinf(x_temp)] = 0
        hxy1 = -np.sum(normed_glcm * x_temp)
        x_temp = np.log(normed_glcm_x)
        x_temp[np.isinf(x_temp)] = 0
        hx = -np.sum(normed_glcm_x * x_temp)
        x_temp = np.log(normed_glcm_y)
        x_temp[np.isinf(x_temp)] = 0
        hy = -sum(normed_glcm_y * x_temp)
        texture_dict.update({'InfMeasureCorr1': (hxy - hxy1) / max(hx, hy)})
    # Вычисление InfMeasureCorr2
    if 'InfMeasureCorr2' in texture_list:
        x_temp1 = np.dot(np.array([normed_glcm_x]).T, np.array([normed_glcm_y]))
        x_temp2 = np.log(x_temp1)
        x_temp2[np.isinf(x_temp2)] = 0
        hxy2 = -np.sum(x_temp1 * x_temp2)
        texture_dict.update({'InfMeasureCorr2': np.sqrt(1 - np.exp(-2. * (hxy2 - hxy)))})
    # Вычисление MaxProb
    if 'MaxProb' in texture_list:
        texture_dict.update({'MaxProb': np.max(normed_glcm)})
    # Вычисление SumAverage
    if 'SumAverage' in texture_list:
        texture_dict.update(
            {'SumAverage': np.sum(np.dot(np.arange(2, 2 * glcm_width + 1), p_X_plus_Y(normed_glcm)))})
        is_sum_average = True
    # Вычисление SumEntropy
    if 'SumEntropy' in texture_list:
        x_temp1 = p_X_plus_Y(normed_glcm)
        x_temp2 = np.log(x_temp1)
        x_temp2[np.isinf(x_temp2)] = 0
        texture_dict.update({'SumEntropy': -np.sum(x_temp1 * x_temp2)})
    # Вычисление SumSquares
    if 'SumSquares' in texture_list:
        if ~is_mui:
            mui = mean_index(I, normed_glcm)
        texture_dict.update({'SumSquares': np.sum(((I - mui) ** 2 * normed_glcm))})
    # Вычисление SumVariance
    if 'SumVariance' in texture_list:
        if is_sum_average:
            f_sum_average = texture_dict['SumAverage']
        else:
            f_sum_average = np.sum(np.dot(np.arange(2, 2 * glcm_width + 1), p_X_plus_Y(normed_glcm)))
        texture_dict.update({'SumVariance': np.sum(
            ((np.array([np.arange(2, 2 * glcm_width + 1)]).T - f_sum_average) ** 2) * p_X_plus_Y(normed_glcm))})
    if 'SDGL' in texture_list:
        texture_dict.update({'SDGL': window.std()})
    return texture_dict

def std_index(IDX, GLCMnorm, IDXmean):
    return np.sqrt(np.sum((IDX - IDXmean) ** 2 * GLCMnorm))


def mean_index(IDX, GLCMnorm):
    return np.sum(IDX * GLCMnorm)


def p_X_minus_Y(P):
    pxmy = np.zeros((1, len(P)))[0]
    for i in range(0, len(P)):
        for j in range(0, len(P)):
            k = abs(i - j)
            pxmy[k] += P[i, j]
    return pxmy


def p_X_plus_Y(P):
    pxpy = np.zeros((2 * len(P) - 1, 1))
    for i in range(0, len(P)):
        for j in range(0, len(P)):
            k = i + j
            pxpy[k] += P[i, j]
    return pxpy


# def my_to_calc_textures(window, I, J, texture_list):
#     mui = 0.0
#     sigi = 0.0
#     muj = 0.0
#     sigj = 0.0
#     hxy = 0.0
#     normed_window_x = 0
#     normed_window_y = 0
#     window_width = len(window)
#     texture_dict = {}
#     window_sum = np.sum(window)
#     if window_sum != 0:
#         normed_window = window / window_sum
#     else:
#         normed_window = window
#     is_sum_average = False
#     is_hxy = False
#     is_mui = False
#     if any(['ClusterProminence' in texture_list, 'Correlation' in texture_list, 'ClusterShade' in texture_list]):
#         mui = mean_index(I, normed_window)
#         is_mui = True
#         sigi = std_index(I, normed_window, mui)
#         muj = mean_index(J, normed_window)
#         sigj = std_index(J, normed_window, muj)
#     if any(['InfMeasureCorr1' in texture_list, 'InfMeasureCorr2' in texture_list]):
#         normed_window_x = np.sum(normed_window, axis=1)
#         normed_window_y = np.sum(normed_window, axis=0)
#
#         x_temp = np.log(normed_window)
#         x_temp[np.isinf(x_temp)] = 0
#         hxy = -np.sum(normed_window * x_temp)
#         is_hxy = True
#     # Вычисление Autocorrelation
#     if 'Autocorrelation' in texture_list:
#         texture_dict.update({'Autocorrelation': np.sum(I * J ** 2 * normed_window)})
#     # Вычисление ClusterProminence
#     if 'ClusterShade' in texture_list:
#         texture_dict.update({'ClusterProminence': np.sum((I + J - mui - muj) ** 4 * normed_window)})
#     # Вычисление ClusterShade
#     if 'ClusterShade' in texture_list:
#         texture_dict.update({'ClusterShade': np.sum((I + J - mui - muj) ** 3 * normed_window)})
#     # Вычисление Contrast
#     if 'Contrast' in texture_list:
#         texture_dict.update({'Contrast': np.sum((I - J) ** 2 * normed_window)})
#     # Вычисление Correlation
#     if 'Correlation' in texture_list:
#         texture_dict.update({'Correlation': np.sum(normed_window * (I - mui) * (J - muj) / (sigi * sigj))})
#     # Вычисление DiffEntropy
#     if 'DiffEntropy' in texture_list:
#         x_temp1 = p_X_minus_Y(normed_window)
#         x_temp2 = np.log(x_temp1)
#         x_temp2[np.isinf(x_temp2)] = 0
#         texture_dict.update({'DiffEntropy': -np.sum(x_temp1 * x_temp2)})
#     # Вычисление DiffVariance
#     if 'DiffVariance' in texture_list:
#         udiff = np.sum(np.arange(0, window_width) * p_X_minus_Y(normed_window))
#         texture_dict.update({'DiffVariance': np.sum(((np.arange(0, window_width) - udiff) ** 2) * p_X_minus_Y(normed_window))})
#     # Вычисление Dissimilarity
#     if 'Dissimilarity' in texture_list:
#         texture_dict.update({'Dissimilarity': np.sum(np.abs(I - J) * normed_window)})
#     # Вычисление Energy
#     if 'Energy' in texture_list:
#         texture_dict.update({'Energy': np.sum(normed_window ** 2)})
#     # Вычисление Entropy
#     if 'Entropy' in texture_list:
#         if is_hxy:
#             texture_dict.update({'Entropy':  hxy})
#         else:
#             x_temp = np.log(normed_window)
#             x_temp[np.isinf(x_temp)] = 0
#             texture_dict.update({'Entropy': -np.sum(normed_window * x_temp)})
#     # Вычисление Homogeneity
#     if 'Homogeneity' in texture_list:
#         texture_dict.update({'Homogeneity': np.sum(normed_window / (abs(I - J) + 1))})
#     # Вычисление Homogeneity2
#     if 'Homogeneity2' in texture_list:
#         texture_dict.update({'Homogeneity2': np.sum(normed_window / ((I - J) ** 2 + 1))})
#     # Вычисление InfMeasureCorr1
#     if 'InfMeasureCorr1' in texture_list:
#         x_temp = np.log(np.dot(np.array([normed_window_x]).T, np.array([normed_window_y])))
#         x_temp[np.isinf(x_temp)] = 0
#         hxy1 = -np.sum(normed_window * x_temp)
#         x_temp = np.log(normed_window_x)
#         x_temp[np.isinf(x_temp)] = 0
#         hx = -np.sum(normed_window_x * x_temp)
#         x_temp = np.log(normed_window_y)
#         x_temp[np.isinf(x_temp)] = 0
#         hy = -sum(normed_window_y * x_temp)
#         texture_dict.update({'InfMeasureCorr1': (hxy - hxy1) / max(hx, hy)})
#     # Вычисление InfMeasureCorr2
#     if 'InfMeasureCorr2' in texture_list:
#         x_temp1 = np.dot(np.array([normed_window_x]).T, np.array([normed_window_y]))
#         x_temp2 = np.log(x_temp1)
#         x_temp2[np.isinf(x_temp2)] = 0
#         hxy2 = -np.sum(x_temp1 * x_temp2)
#         texture_dict.update({'InfMeasureCorr2': np.sqrt(1 - np.exp(-2. * (hxy2 - hxy)))})
#     # Вычисление MaxProb
#     if 'MaxProb' in texture_list:
#         texture_dict.update({'MaxProb': np.max(normed_window)})
#     # Вычисление SumAverage
#     if 'SumAverage' in texture_list:
#         texture_dict.update({'SumAverage': np.sum(np.dot(np.arange(2, 2 * window_width + 1), p_X_plus_Y(normed_window)))})
#         is_sum_average = True
#     # Вычисление SumEntropy
#     if 'SumEntropy' in texture_list:
#         x_temp1 = p_X_plus_Y(normed_window)
#         x_temp2 = np.log(x_temp1)
#         x_temp2[np.isinf(x_temp2)] = 0
#         texture_dict.update({'SumEntropy': -np.sum(x_temp1 * x_temp2)})
#     # Вычисление SumSquares
#     if 'SumSquares' in texture_list:
#         if ~is_mui:
#             mui = mean_index(I, normed_window)
#         texture_dict.update({'SumSquares': np.sum(((I - mui) ** 2 * normed_window))})
#     # Вычисление SumVariance
#     if 'SumVariance' in texture_list:
#         if is_sum_average:
#             f_sum_average = texture_dict['SumAverage']
#         else:
#             f_sum_average = np.sum(np.dot(np.arange(2, 2 * window_width + 1), p_X_plus_Y(normed_window)))
#         texture_dict.update({'SumVariance': np.sum(((np.array([np.arange(2, 2 * window_width + 1)]).T - f_sum_average) ** 2) * p_X_plus_Y(normed_window))})
#     return texture_dict
