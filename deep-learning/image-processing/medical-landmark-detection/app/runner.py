# --------------------------------------------------------------------------- #
# IMPORTS
# --------------------------------------------------------------------------- #

import os
import math

import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import random
import shutil
import SimpleITK as sitk
import numpy as np
import pandas as pd
from scipy import stats, ndimage

import torch
from torch.utils.data import DataLoader

from .utils import *
from .datasets import get_dataset
from .model import get_loss, get_optim, get_net, get_scheduler


# --------------------------------------------------------------------------- #
# CLASS DEFINITION
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
class Runner(object):

    def __init__(self, args):

        self.args = args
        self.phase = self.args.phase

    def setup_seed(selfself, seed):

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def get_opts(self):

        self.opts = get_config(self.args.config)
        update_config(self.opts, self.args)
        self.run_name = self.opts.run_name if self.opts.run_name else self.opts.model
        self.run_dir = os.path.join(self.opts.run_dir, self.run_name)
        d_opts = self.opts.dataset
        self.name_list = d_opts['name_list']
        self.run_dir = os.path.join(self.run_dir, '-'.join(self.name_list))
        mkdir(self.run_dir)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opts.cuda_devices
        self.setup_seed(self.opts.seed)

    def get_loader(self):

        def _get(s='train'):
            dataset_list = []
            loader_list = []
            trans_dic = self.opts['transform_params'] if s == 'train' else {}
            for name in self.name_list:
                kwargs_dataset = {k: v for k, v in d_opts[name].items() \
                                  if k in [arg for arg in get_dataset(name).__init__.__code__.co_varnames \
                                           if arg not in ['phase', 'use_background_channel']]}
                d = get_dataset(name)(phase=s,
                                      transform_params=trans_dic,
                                      use_background_channel=use_background_channel,
                                      **kwargs_dataset)
                dataset_list.append(d)
                loader_opts = self.opts.dataloader[s]
                kwargs_dataset = {k: v for k, v in loader_opts.items() \
                                  if k in [arg for arg in DataLoader.__init__.__code__.co_varnames]}
                if s == 'train' and 'batch_size_dic' in d_opts:
                    loader_opts['batch_size'] = d_opts['batch_size_dic'][name]
                l = DataLoader(d, **kwargs_dataset)
                loader_list.append(l)
            setattr(self, s + '_dataset_list', dataset_list)
            setattr(self, s + '_loader_list', loader_list)

        d_opts = self.opts.dataset
        use_background_channel = self.opts.use_background_channel
        self.loss_weights = d_opts['loss_weights']
        # self.name_list = d_opts['name_list']
        # self.run_dir = os.path.join(self.run_dir, '-'.join(self.name_list))
        # mkdir(self.run_dir)
        to_yaml(f'{self.run_dir}/config_{self.phase}.yaml', self.opts)
        shutil.copy(self.args.config, f'{self.run_dir}/config_origin.yaml')
        self.train_name_list = d_opts['name_list']
        if self.phase == 'train':
            _get('train')
            _get('validate')
            step = int(self.opts.mix_step)
            if step > 0:
                self.train_name_list = ['mix']
                self.train_loader_list = [MixIter(self.train_loader_list, step)]
        elif self.phase == 'test':
            _get('test')
        else:
            _get('validate')

    def get_model(self):

        def get_learner():
            learn = self.opts.learning
            loss_args = {k: v for k, v in learn[learn['loss']].items() \
                         if k in [arg for arg in get_loss(learn['loss']).__init__.__code__.co_varnames]}
            self.loss = get_loss(learn['loss'])(**loss_args)
            self.val_loss = get_loss(learn['loss'])(**loss_args)
            optim_args = {k: v for k, v in learn[learn['optim']].items() \
                          if k in [arg for arg in get_optim(learn['optim']).__init__.__code__.co_varnames]}
            self.optim = get_optim(learn['optim'])(self.model.parameters(), **optim_args)
            if learn['use_scheduler']:
                scheduler_args = {k: v for k, v in learn[learn['scheduler']].items() \
                                  if
                                  k in [arg for arg in get_scheduler(learn['scheduler']).__init__.__code__.co_varnames]}
                self.scheduler = get_scheduler(learn['scheduler'])(self.optim, **scheduler_args)
            else:
                self.scheduler = None

        modelname = self.opts.model
        model_opts = self.opts[modelname] if modelname in self.opts.keys() else {}
        localNet = model_opts['localNet'] if 'localNet' in model_opts else None
        dataset = self.opts['dataset']
        channel_params = {'in_channels': [], 'out_channels': []}
        is_2d_flags = []
        img_size_list = []
        for name in dataset['name_list']:
            size = dataset[name]['size']
            is_2d_flags.append(len(size) == 2)
            img_size_list.append(size)
            channel_params['in_channels'].append(1)
            channel_params['out_channels'].append(dataset[name]['num_landmark'])
        if self.opts.use_background_channel:
            li = channel_params['out_channels']
            for i in range(len(li)):
                li[i] += 1

        globalNet_params = channel_params.copy()
        localNet_params = channel_params.copy()

        if modelname.startswith('gln'):
            globalNet_params_final = model_opts['globalNet_params']
            for k, v in globalNet_params.items():
                globalNet_params_final[k] = v
            self.model = get_net(modelname)(get_net(localNet),
                                            localNet_params,
                                            globalNet_params_final)
        else:
            net_params = self.opts[modelname] if modelname in self.opts.keys() else {}
            net_params.update(channel_params)
            self.model = get_net(modelname)(**net_params)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        if os.path.isfile(self.opts.checkpoint):
            print(f"#    -> trained model loaded: {self.opts.checkpoint}")
            checkpoint = torch.load(self.opts.checkpoint)
            self.start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['model_state_dict'])
            get_learner()
        else:
            self.start_epoch = 0
            get_learner()
        with open(os.path.join(self.run_dir, 'network_architecture.txt'), 'w') as f:
            f.write(str(self.model))
        if torch.cuda.is_available() and self.phase == 'train':
            self.loss = self.loss.cuda()
            self.val_loss = self.val_loss.cuda()
        self.device = next(self.model.parameters()).device

    def config(self):

        self.get_opts()
        self.get_loader()
        self.get_model()

    def update_params(self, pbar):

        self.model.train()
        self.train_loss = 0
        use_scheduler = self.opts.learning.use_scheduler
        for task_idx, (name, loader) in enumerate(zip(self.train_name_list, self.train_loader_list)):
            batch_num = len(loader)
            cur_loss = 0
            for i, data_dic in enumerate(loader):
                if isinstance(data_dic, tuple):
                    task_idx = data_dic[1]
                    data_dic = data_dic[0]
                for k in {'input', 'gt'}:
                    data_dic[k] = torch.autograd.Variable(data_dic[k]).to(self.device)
                data_dic.update(self.model(data_dic['input'], task_idx))
                self.optim.zero_grad()
                loss = self.loss(data_dic['output'], data_dic['gt'])
                if 'rec_image' in data_dic:
                    loss += get_loss('l2')(**self.opts.learning.l2)(data_dic['input'],
                                                                    data_dic['rec_image'])
                if hasattr(self, 'loss_weights'):
                    loss *= self.loss_weights[task_idx]
                cur_loss += loss.item()
                loss.backward()
                self.lr_list.append(self.optim.param_groups[0]['lr'])
                self.optim.step()
                if use_scheduler:
                    self.scheduler.step()
                pbar.set_description(f"[{'train':8s}] "
                                     f"({(i + 1):>3d}/{batch_num:<3d}) "
                                     f"train_loss: {loss.item():10.2f} | "
                                     f"{'best_eval_loss:':15s} {self.best_loss:10.2f}")
            self.train_loss += cur_loss / len(loader)

    def validate(self, epoch=None):

        self.model.eval()
        if epoch is None:
            epoch = self.start_epoch
        prefix = self.run_dir + '/results/' + self.phase + f'_epoch{epoch + 1:03d}'
        loss_dir = self.run_dir + '/results/loss'
        mkdir(loss_dir)
        if self.phase != 'train':
            mkdir(prefix)
        s = 'validate'
        loader_list = getattr(self, 'validate_loader_list')
        val_loss = 0
        for task_idx, (name, cur_loader) in enumerate(zip(self.name_list, loader_list)):
            best = os.path.join(prefix, name)
            if self.phase != 'train':
                mkdir(best)
            batch_num = len(cur_loader)
            pbar = tqdm(enumerate(cur_loader),
                        unit='epoch')
            name_loss_dic = {}
            for i, data_dic in pbar:
                for k in {'input', 'gt'}:
                    if k in data_dic:
                        data_dic[k] = torch.autograd.Variable(data_dic[k].to(self.device))
                with torch.no_grad():
                    data_dic.update(self.model(data_dic['input'], task_idx))
                if 'gt' in data_dic:
                    if data_dic['output'].shape != data_dic['gt'].shape:
                        print(data_dic['path'])
                        exit()
                    loss = self.val_loss(data_dic['output'], data_dic['gt'])
                    if 'rec_image' in data_dic:
                        loss += get_loss('l2')(**self.opts.learning.l2)(data_dic['input'],
                                                                        data_dic['rec_image'])
                    pbar.set_description(f"[{s.ljust(8)}] "
                                         f"({(i + 1):>3d}/{batch_num:<3d}) "
                                         f"train_loss: {self.train_loss:10.2f} | "
                                         f"{'val_loss:':15s} {loss.item():10.2f}")
                    name_loss_dic['_'.join(data_dic['name'])] = loss.item()
            if 'gt' in data_dic:
                mean = np.mean(list(name_loss_dic.values()))
                val_loss += mean
                path = os.path.join(loss_dir, f'epoch_{epoch + 1:03d}_val_loss_{mean:.2f}.txt')
                with open(path, 'w') as f:
                    for k, v in name_loss_dic.items():
                        f.write(f"{v:.6f} {mean}\n")

        return val_loss

    def train(self):

        self.model.train()
        checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
        mkdir(checkpoint_dir)
        pbar = tqdm(range(self.start_epoch, self.opts.epochs),
                    unit='epoch')
        xs, ys = [], []
        self.lr_list = []
        loss_file = os.path.join(self.run_dir, 'train_val_loss_history.csv')
        end_epoch = self.opts.epochs - 1
        save_freq = self.opts.save_freq
        eval_freq = self.opts.eval_freq

        for epoch in pbar:
            self.update_params(pbar)

            if epoch % eval_freq == 0 or epoch == end_epoch:
                val_loss = self.validate(epoch)
                xs.append(epoch)
                ys.append(val_loss)
                data = {'epoch': epoch,
                        'model_name': self.run_name,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer': self.optim,
                        'scheduler': self.scheduler}
                save_name = f'{self.run_name}_epoch{epoch + 1:03d}_train{self.train_loss:.2f}_val{val_loss:.2f}.pt'
                if (save_freq != 0 and epoch % save_freq == 0) or epoch == end_epoch:
                    torch.save(data, os.path.join(checkpoint_dir, save_name))
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    best = os.path.join(checkpoint_dir, 'best_' + save_name)
                    self.opts.checkpoint = best
                    torch.save(data, best)
            with open(loss_file, 'a') as f:
                if epoch % eval_freq == 0 or epoch == end_epoch:
                    f.write(f"{epoch + 1:03d},{self.train_loss},{val_loss}\n")
                else:
                    f.write(f"{epoch + 1:03d},{self.train_loss},{''}\n")
        plot_2d(self.run_dir + '/learning_rate_steps.png',
                list(range(len(self.lr_list))),
                self.lr_list,
                xlabel='step',
                ylabel='learning rate',
                title='Learning rate steps')
        plot_2d(self.run_dir + '/validation_loss_history.png',
                xs,
                ys,
                xlabel='epoch',
                ylabel='loss',
                title='Validation loss history')

    def test(self, epoch=None):

        self.model.eval()
        if epoch is None:
            epoch = self.start_epoch
        loader_list = getattr(self, f'{self.phase}_loader_list')
        test_loss = 0
        for task_idx, (name, cur_loader) in enumerate(zip(self.name_list, loader_list)):
            best = os.path.join(self.run_dir, 'results', self.phase + f'_epoch{epoch + 1:03d}')
            loss_dir = os.path.join(self.run_dir, 'results/loss')
            mkdir(best)
            mkdir(loss_dir)
            batch_num = len(cur_loader)
            pbar = tqdm(enumerate(cur_loader), unit='epoch')
            name_loss_dic = {}
            landmarks_dic = {}
            dist_err_dic = {}
            for i, data_dic in pbar:
                for k in {'input', 'gt'}:
                    if k in data_dic:
                        data_dic[k] = torch.autograd.Variable(data_dic[k].to(self.device))
                with torch.no_grad():

                    # GROUND TRUTH
                    # Format ground truth heatmap to numpy array
                    ground_truth = data_dic['gt'].detach().cpu().numpy()
                    # Calculate coordinates of each ground truth landmark
                    gt_pts = []
                    for idx in range(ground_truth.shape[1]):
                        gt_pts.append(np.unravel_index(ground_truth[0, idx, :, :].argmax(),
                                                       ground_truth[0, idx, :, :].shape))
                    # Combine all frames from ground truth array (i.e. one frame per landmark)
                    ground_truth = np.nansum(ground_truth[0, :, :, :], axis=0)
                    # Apply threshold to isolate local maxima in model generated heatmap
                    ground_truth[ground_truth <= 0.01] = np.nan

                    # MODEL PREDICTION
                    # Generate model output (predictions)
                    data_dic.update(self.model(data_dic['input'], task_idx))
                    # Extract model output information
                    model_output = data_dic['output'].detach().cpu().numpy()
                    # Combine all frames from model output (i.e. one frame per landmark)
                    model_output = np.nansum(model_output[0, :, :, :], axis=0)
                    # Apply threshold to isolate local maxima in model generated heatmap
                    model_output[model_output <= 0.05] = np.nan
                    # Apply maximum filter to detect coordinates of local maxima
                    maxima = (model_output == ndimage.filters.maximum_filter(model_output,
                                                                             3,
                                                                             mode='constant',
                                                                             cval=0.0))
                    # Find coordinates of local maxima (i.e. predicated landmarks)
                    predict_pts = np.argwhere(maxima == True)
                    # Sort predictions along the y-axis to match with ground truth order
                    predict_pts = predict_pts[predict_pts[:, 1].argsort()]

                    # Initialize distance array
                    dist_err = []
                    # Loop through each landmark
                    for land_idx, (gt, pr) in enumerate(zip(gt_pts, predict_pts), start=1):
                        # Add landmarks coordinates to landmarks dictionary
                        landmarks_dic[data_dic['name'][0] + f'_l{land_idx}'] = [gt[0], pr[0], gt[1], pr[1]]
                        # Calculate distance between model output and ground truth
                        dist_err.append(math.dist(gt, pr))
                    dist_err_dic[data_dic['name'][0]] = dist_err

                    # Generate and save figures
                    plt.figure(figsize=(12, 12))
                    plt.imshow(data_dic['input'].detach().cpu().numpy()[0, 0, :, :], cmap='gray')
                    plt.imshow(ground_truth, cmap='YlOrRd')
                    plt.axis('off')
                    plt.savefig(best + '/' + data_dic['name'][0] + '_gt_heatmap.png',
                                bbox_inches='tight', pad_inches=0)
                    plt.close()
                    plt.figure(figsize=(12, 12))
                    plt.imshow(data_dic['input'].detach().cpu().numpy()[0, 0, :, :], cmap='gray')
                    plt.imshow(model_output, cmap='Blues')
                    plt.axis('off')
                    plt.savefig(best + '/' + data_dic['name'][0] + '_pred_heatmaps.png',
                                bbox_inches='tight', pad_inches=0)
                    plt.close()
                    plt.figure(figsize=(12, 12))
                    plt.imshow(data_dic['input'].detach().cpu().numpy()[0, 0, :, :], cmap='gray')
                    for gt, pr in zip(gt_pts, predict_pts):
                        plt.plot(gt[1], gt[0], marker='x', markersize=30, c='gold')
                        plt.plot(pr[1], pr[0], marker='x', markersize=30, c='dodgerblue')
                    plt.axis('off')
                    plt.savefig(best + '/' + data_dic['name'][0] + '_centroids.png',
                                bbox_inches='tight', pad_inches=0)
                    plt.close()

                # save_data(data_dic)
                if 'gt' in data_dic:
                    if data_dic['output'].shape != data_dic['gt'].shape:
                        print(data_dic['path'])
                        exit()
                    loss = self.val_loss(data_dic['output'], data_dic['gt'])
                    if 'rec_image' in data_dic:
                        loss += get_loss('l2')(**self.opts.learning.l2)(data_dic['input'],
                                                                        data_dic['rec_image'])
                    pbar.set_description(f"[{self.phase.ljust(8)}] "
                                         f"({(i + 1):>3d}/{batch_num:<3d}) "
                                         f"train_loss: {self.train_loss:10.2f} | "
                                         f"{'test_loss:':15s} {loss.item():10.2f}")
                    name_loss_dic['_'.join(data_dic['name'])] = loss.item()
            if 'gt' in data_dic:
                mean = np.mean(list(name_loss_dic.values()))
                test_loss += mean
                path = os.path.join(loss_dir, f'epoch_{epoch + 1:03d}_test_loss_{mean:.2f}.txt')
                with open(path, 'w') as f:
                    for k, v in name_loss_dic.items():
                        f.write(f"{v:.6f} {mean}\n")

            # Generate and export landmarks coordinates dataframe
            landmarks_dic['r'] = [np.nan] * 4
            landmarks_dic['r2'] = [np.nan] * 4
            landmarks_df = pd.DataFrame.from_dict(landmarks_dic,
                                                  orient='index',
                                                  columns=['x_gt', 'x_pred', 'y_gt', 'y_pred'])
            x_r, _ = stats.pearsonr(landmarks_df['x_gt'].iloc[:-2],
                                    landmarks_df['x_pred'].iloc[:-2])
            y_r, _ = stats.pearsonr(landmarks_df['y_gt'].iloc[:-2],
                                    landmarks_df['y_pred'].iloc[:-2])
            landmarks_df.loc['r']['x_pred'] = x_r
            landmarks_df.loc['r']['y_pred'] = y_r
            landmarks_df.loc['r2']['x_pred'] = x_r ** 2
            landmarks_df.loc['r2']['y_pred'] = y_r ** 2
            landmarks_df.to_csv(best + '/coordinate_results.csv')
            # Generate and export distance DataFrame
            dist_err_dic['mean'] = [np.nan] * len(dist_err_dic[list(dist_err_dic)[0]])
            dist_err_dic['sd'] = [np.nan] * len(dist_err_dic[list(dist_err_dic)[0]])
            dist_err_df = pd.DataFrame.from_dict(dist_err_dic,
                                                 orient='index',
                                                 columns=[f'l{i}' for i in
                                                          range(len(dist_err_dic[list(dist_err_dic)[0]]))])
            dist_err_df.loc['mean'] = [np.nanmean(dist_err_df[f'l{i}'].iloc[:-2]) for i in
                                       range(len(dist_err_dic[list(dist_err_dic)[0]]))]
            dist_err_df.loc['sd'] = [np.nanstd(dist_err_df[f'l{i}'].iloc[:-2]) for i in
                                     range(len(dist_err_dic[list(dist_err_dic)[0]]))]
            dist_err_df.to_csv(best + '/distance_results.csv')

        return test_loss

    def run(self):

        self.config()
        self.best_loss = self.train_loss = float('inf')
        if self.phase == 'train':
            print(f"# Model Training in Progress [selected model: {self.opts.model}] ")
            print(f"    -> model architecture selected: {self.opts.model} ")
            self.train()
            print("# Training Completed")
            print("#-----------------------------------------------------------------------------#")
            self.phase = 'test'
            print(f"# Model Testing in Progress")
            self.get_loader()
            self.get_model()
            self.test(self.start_epoch)
            print("# Testing Completed")
            print("#-----------------------------------------------------------------------------#")
        else:
            self.test(self.start_epoch)
        print("# Program Completed")
        print("#-----------------------------------------------------------------------------#")
