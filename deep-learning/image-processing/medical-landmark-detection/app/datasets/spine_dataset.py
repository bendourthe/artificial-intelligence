# --------------------------------------------------------------------------- #
# IMPORTS
# --------------------------------------------------------------------------- #

import os
from PIL import Image

import numpy as np
import pandas as pd
import ast
import torch
import torch.utils.data as data

from ..utils import gaussian_heatmap, transformer


# --------------------------------------------------------------------------- #
# CLASS DEFINITION
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
class Spine(data.Dataset):

    def __init__(self,
                 prefix,
                 phase,
                 transform_params={},
                 sigma=5,
                 num_landmark=6,
                 size=[512, 1024],
                 exclude_list=None,
                 use_background_channel=False):
        self.transform = transformer(transform_params)
        self.size = tuple(size)
        self.num_landmark = num_landmark
        self.use_background_channel = use_background_channel
        self.path_img = os.path.join(prefix, 'images')
        self.path_lab = os.path.join(prefix, 'labels')
        self.labels_df = pd.read_csv(os.path.join(self.path_lab,
                                                  'labels.csv'),
                                     index_col=0)
        files = [file[:-4] for file in sorted(os.listdir(self.path_img))]
        if exclude_list is not None:
            st = set(exclude_list)
            files = [f for f in files if f not in st]
        n = len(files)
        train_num = int(0.80*n)
        val_num = int(0.10*n)
        test_num = n - train_num - val_num
        if phase == 'train':
            self.indices = files[:train_num]
        elif phase == 'validate':
            self.indices = files[train_num:-test_num]
        elif phase == 'test':
            self.indices = files[-test_num:]
        else:
            raise Exception(f'Unknown phase: {phase}')
        self.gen_heatmap = gaussian_heatmap(sigma, dim=len(size))

    def __getitem__(self, index):
        name = self.indices[index]
        ret = {'name': name}
        img, origin_size = self.read_image(os.path.join(self.path_img,
                                                        name + '.png'))
        self.origin_size = origin_size
        labels = self.read_labels_csv(name)
        li = [self.gen_heatmap(lab, self.size) for lab in labels]
        if self.use_background_channel:
            sm = sum(li)
            sm[sm > 1] = 1
            li.append(1 - sm)
        gt = np.array(li)
        img, gt = self.transform(img, gt)
        ret['input'] = torch.FloatTensor(img)
        ret['gt'] = torch.FloatTensor(gt)
        return ret

    def __len__(self):

        return len(self.indices)

    def read_image(self, path):
        img = Image.open(path)
        origin_size = img.size
        img = img.resize(self.size)
        arr = np.array(img)
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = np.expand_dims(np.transpose(arr, (1, 0)), 0).astype(np.float)
        for i in range(arr.shape[0]):
            arr[i] = (arr[i] - arr[i].mean()) / (arr[i].std() + 1e-20)
        return arr, origin_size

    def read_labels_csv(self, name):
        labels = self.labels_df.loc[name].apply(lambda x: ast.literal_eval(str(x))).values
        res_labels = []
        for lab in labels:
            res_labels.append((int(lab[0] * self.size[0]/self.origin_size[0]),
                               int(lab[1] * self.size[1]/self.origin_size[1])))
        return res_labels
