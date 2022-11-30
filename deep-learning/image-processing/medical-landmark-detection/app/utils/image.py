# --------------------------------------------------------------------------- #
# IMPORTS
# --------------------------------------------------------------------------- #

import os
import SimpleITK as sitk
import numpy as np


# --------------------------------------------------------------------------- #
# METHODS DEFINITION
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
def coord2index(coord, origin, direction, spacing):
    D = np.matrix(direction).reshape(3, 3)
    S = np.diag(spacing)
    m = (D * S).I

    index = ((np.array(coord) - origin) * m).getA().flatten().tolist()

    return tuple(round(i) for i in index)[::-1]


# --------------------------------------------------------------------------- #
def index2coord(index, origin, direction, spacing):
    index = np.array(index[::-1])
    D = np.matrix(direction).reshape(3, 3)
    S = np.diag(spacing)
    m = (D * S)
    coord = (index * m).getA().flatten() + origin

    return tuple(coord.tolist())


# --------------------------------------------------------------------------- #
def get_info(itk):
    info = {}
    inf['direction'] = itk.GetDirection()
    info['origin'] = itk.GetOrigin()
    info['spacing'] = itk.GetSpacing()

    return itk


# --------------------------------------------------------------------------- #
def set_info(itk, info):
    itk.SetDirection(info['direction'])
    itk.SetOrigin(info['origin'])
    itk.SetSpacing(info['spacing'])

    return itk


# --------------------------------------------------------------------------- #
def get_flip(dires, target=(1, 0, 0, 0, -1, 0, 0, 0, -1)):
    is_right = [True,
                dires[8] * target[8] > 0,
                dires[4] * target[4] > 0,
                dires[0] * target[0] > 0]

    return tuple(slice(None, None, 2 * i - 1) for i in is_right)


# --------------------------------------------------------------------------- #
def read_ITK(path):
    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        itk = reader.Execute()
    else:
        itk = sitk.ReadImage(path)

    return itk
