# --------------------------------------------------------------------------- #
# IMPORTS
# --------------------------------------------------------------------------- #

import numpy as np
from skimage import transform as sktransform


# --------------------------------------------------------------------------- #
# METHODS DEFINITION
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
def rotate(angle):
    def fn(img):
        ret = []
        for i in range(img.shape[0]):
            ret.append(sktransform.rotate(img[i], angle))

        return np.array(ret)

    return fn


# --------------------------------------------------------------------------- #
def translate(offsets):
    offsets = tuple(offsets)
    new_sls = tuple(slice(i, None) for i in offsets)

    def fn(img):
        ret = []
        size = img.shape[1:]
        old_sls = tuple(slice(0, j - i) for i, j in zip(offsets, size))
        for old in img:
            new = np.zeros(size)
            new[new_sls] = old[old_sls]
            ret.append(new)

        return np.array(ret)

    return fn


# --------------------------------------------------------------------------- #
def flip(axis=1):
    f_sls = slice(None, None, -1)
    sls = slice(None, None)

    def fn(img):
        dim = img.ndim
        cur_axis = axis % dim
        if cur_axis == 0:
            all_sls = tuple([f_sls]) * dim
        else:
            all_sls = tuple(f_sls if i == cur_axis else sls for i in range(dim))

            return img[all_sls]

    return fn


# --------------------------------------------------------------------------- #
def transformer(param_dict):
    fs = []
    if 'flip_rate' in param_dict \
            and np.random.rand() < param_dict['flip_rate']:
        fs.append(flip(param_dict['axis']))
    if 'rotate_rate' in param_dict \
            and np.random.rand() < param_dict['rotate_rate']:
        fs.append(rotate(param_dict['angle']))
    if 'translate_rate' in param_dict \
            and np.random.rand() < param_dict['translate_rate']:
        fs.append(translate(param_dict['offsets']))

    def trans(*imgs):
        ret = []
        for img in imgs:
            cur_img = img.copy()
            for f in fs:
                cur_img = f(cur_img)
            ret.append(cur_img.copy())

        return tuple(ret)

    return trans
