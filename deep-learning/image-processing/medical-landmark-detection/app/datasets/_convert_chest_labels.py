# --------------------------------------------------------------------------- #
# IMPORTS
# --------------------------------------------------------------------------- #

import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# FUNCTION DEFINITION
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
def read_labels_txt(path, filename, size_):
    path = os.path.join(path, filename)
    labels = []
    with open(path, 'r') as f:
        n = int(f.readline())
        for i in range(n):
            ratios = [float(i) for i in f.readline().split()]
            pt = tuple([round(r * sz) for r, sz in zip(ratios, size_)])
            labels.append(pt)
    return labels


# --------------------------------------------------------------------------- #
def convert_txt_labs_to_csv(path_lab, size_):
    csv_path = os.path.join(path_lab, 'labels.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    lab_files = sorted(os.listdir(path_lab))
    labs_list = []
    for lab_file in lab_files:
        labs = read_labels_txt(path_lab, lab_file, size_)
        #labs_list.append(list(sum(labs, ())))
        labs_list.append(labs)
    # labs_df = pd.DataFrame(data=labs_list,
    #                        index=[fn[:-4] for fn in lab_files],
    #                        columns=(['l1_x', 'l1_y',
    #                                  'l2_x', 'l2_y',
    #                                  'l3_x', 'l3_y',
    #                                  'l4_x', 'l4_y',
    #                                  'l5_x', 'l5_y',
    #                                  'l6_x', 'l6_y']))
    labs_df = pd.DataFrame(data=labs_list,
                           index=[fn[:-4] for fn in lab_files],
                           columns=(['l1', 'l2', 'l3', 'l4', 'l5', 'l6']))
    labs_df.to_csv(csv_path)

# --------------------------------------------------------------------------- #
# LABELS CONVERSION
# --------------------------------------------------------------------------- #

size_ = tuple([512, 512])
path_lab = '../../data/chest/labels'

convert_txt_labs_to_csv(path_lab, size_)
import ast
labels = pd.read_csv(os.path.join(path_lab, 'labels.csv'), index_col=0)
print(labels.loc['CHNCXR_0001_0'].apply(lambda x: ast.literal_eval(str(x))).values)


# --------------------------------------------------------------------------- #
def read_image(path, filename, size_):
    img = Image.open(os.path.join(path, filename))
    origin_size = img.size
    img = img.resize(size_)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.expand_dims(np.transpose(arr, (1, 0)), 0).astype(float)
    for i in range(arr.shape[0]):
        arr[i] = (arr[i] - arr[i].mean()) / (arr[i].std() + 1e-20)
    return arr, origin_size


path_img = '../../data/chest/images'
img_files = sorted(os.listdir(path_img))
# for i, (img_file, lab_file) in enumerate(zip(img_files, lab_files)):
#     img, origin_size = read_image(path_img, img_file)
#     labs = read_labels(path_lab, lab_file)
#     print(labs)
#     plt.figure(figsize=(6,6))
#     plt.imshow(img[0, :, :], cmap='gray')
#     for pt in labs:
#         plt.plot(pt[0], pt[1], marker='x', ms=10, c='white')
#     plt.title(f'{img_file[:-4]}')
#     plt.show()
#     if i >= 10:
#         break
