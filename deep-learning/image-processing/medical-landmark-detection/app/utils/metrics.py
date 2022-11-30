# --------------------------------------------------------------------------- #
# IMPORTS
# --------------------------------------------------------------------------- #

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import mean_squared_error as MSE
from .kit import norm, get_points_from_heatmap


# --------------------------------------------------------------------------- #
# METHODS DEFINITION
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
def get_metric(s):
    return {'ssim': cal_ssim,
            'psnr': cal_psnr,
            'mse': cal_mse,
            'mre': cal_mre,
            'std': cal_std}[s]


# --------------------------------------------------------------------------- #
def prepare(x):
    if np.iscomplexobj(x):
        x = np.abs(x)

    return norm(x)


# --------------------------------------------------------------------------- #
def cal_mse(x, y):
    x = prepare(x)
    y = prepare(y)

    return MSE(x, y)


# --------------------------------------------------------------------------- #
def cal_ssim(x, y):
    x = prepare(x)
    y = prepare(y)

    return SSIM(x, y, data_range=x.max() - x.min())


# --------------------------------------------------------------------------- #
def cal_psnr(x, y):
    x = prepare(x)
    y = prepare(y)

    return PSNR(x, y, data_range=x.max() - x.min())


# --------------------------------------------------------------------------- #
def cal_mre(x, y):
    p1 = getPointsFromHeatmap(x)
    p2 = getPointsFromHeatmap(y)

    li = [sum((i - j) ** 2 for i, j in zip(point, gt_point)) ** 0.5 for point,
                                                                        gt_point in zip(p1, p2)]

    return np.mean(li)


# --------------------------------------------------------------------------- #
def cal_std(x, y):
    p1 = getPointsFromHeatmap(x)
    p2 = getPointsFromHeatmap(y)

    li = [sum((i - j) ** 2 for i, j in zip(point, gt_point)) ** 0.5 for point,
                                                                        gt_point in zip(p1, p2)]

    return np.std(li)
