from Architectures.AtomCloudNet import *
from Processing.loader2 import *
from utils import *

from tqdm import tqdm, trange
from configparser import ConfigParser
import numpy as np
import ast
import os
import pickle
from tensorboardX import SummaryWriter  # https://sapanachaudhary.github.io/Colab-pages/x

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from se3cnn.util.default_dtype import torch_default_dtype

TRAIN_PATH = 'data/QM9Train'
TEST_PATH = 'data/QM9Test'


class ACN:
    def __init__(self,
                 run_id=1):
        path = 'config_ACN/config_'
        self.hyperparams = get_config2(run_id=run_id, path=path)
        self.type = self.hyperparams[0]
        print(self.hyperparams)
        self.run_id = run_id
        self.verbose = 1
        self.val_size = .05
        self.val_path = 'runs/run_{}/val_{}'.format(int(self.run_id) // 100, int(self.run_id))
        self.train_path = 'runs/run_{}/train_{}'.format(int(self.run_id) // 100, int(self.run_id))
        self.checkpoint_folder = 'models/run_{}/'.format(int(self.run_id) // 100)
        self.save_path = 'models/run_{}/model_{}'.format(int(self.run_id) // 100, int(self.run_id))

        if not os.path.isdir('runs'):
            os.mkdir('runs')
        if not os.path.isdir('models'):
            os.mkdir('models')
        if not os.path.isdir(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)
        if not os.path.isdir(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)
        if not os.path.isdir(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)

        self.val_writer = SummaryWriter(self.val_path)
        self.train_writer = SummaryWriter(self.train_path)

        self.train_path = TRAIN_PATH
        self.test_path = TEST_PATH
        self.batch_size = self.hyperparams[7]
        self.ngpus = 0


        train_data = qm9_loader(limit=10000, path=self.train_path + '/*.xyz', type=self.type, init = False)
        print(train_data.max_two)
        print(train_data.max_three)
        # test_data = qm9_loader(limit=np.inf, path=self.test_path + '/*.xyz', type=self.type, init = True)
        print("\nTotal number of training samples assembled:", train_data.__len__())

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(self.val_size * num_train))

        train_idx, val_idx = indices[split:], indices[:split]
        print("Validation split with {} training samples and {} validation samples. ({} split)."
              .format(len(train_idx), len(val_idx), self.val_size))
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, sampler=train_sampler)
        self.val_loader = DataLoader(train_data, batch_size=self.batch_size, sampler=val_sampler)
        # self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Using device:", self.device)

        self.nepochs = self.hyperparams[6]

    def train_molecular_model(self):
        use_Z_emb = False
        use_23_body = False
        if self.type == 1:
            use_Z_emb = True
        if self.type == 2:
            use_23_body = True
        if self.type == 3:
            use_Z_emb = True
            use_23_body = True
        with torch_default_dtype(torch.float64):
            print("Generating model of type {}".format(self.type))
            model = se3ACN(device=self.device, nclouds=self.hyperparams[3], natoms=30,
                           resblocks=self.hyperparams[5], cloud_dim=self.hyperparams[4],
                           neighborradius=self.hyperparams[2],
                           nffl=self.hyperparams[8], ffl1size=self.hyperparams[9], emb_dim=self.hyperparams[10],
                           cloudord=self.hyperparams[11], two_three=use_23_body, Z=use_Z_emb)
        print("applying weights")
        model.apply(weights_init)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = torch.nn.DataParallel(model)
        model.to(self.device)
        criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        opt = torch.optim.Adam(model.parameters(), lr=self.hyperparams[1])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=7, verbose=True)
        model.train()
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        if self.verbose > 0:
            print(model)
            print("Number trainable parameters:", params)

        train_step = 0
        val_step = 0
        train_pbar = tqdm(self.train_loader)
        val_pbar = tqdm(self.val_loader)

        min_loss = 10000
        for epoch in range(self.nepochs):

            # TRAINING
            tot_loss = 0
            model.train()
            for i, (xyz, prot_ids, features, urt) in enumerate(train_pbar, start=1):
                train_step += self.batch_size
                xyz = xyz.to(self.device)
                featZ = prot_ids.view(xyz.shape[0], xyz.shape[1]).to(self.device)
                if self.type == 1:
                    prediction = model(xyz, featZ, None)
                elif self.type == 2:
                    feat23 = features.to(self.device)
                    prediction = model(xyz, None, feat23)
                elif self.type == 3:
                    feat23 = features.to(self.device)
                    prediction = model(xyz, featZ, feat23)
                target = urt.to(self.device).double()
                loss = criterion(prediction, target)
                mae = mae_criterion(prediction, target)
                opt.zero_grad()
                loss.backward()  # can also be loss = rmse-loss
                opt.step()
                mae_loss = mae.cpu().item()
                loss_ = torch.sqrt(loss).cpu().item()
                tot_loss += mae_loss
                avg_loss = tot_loss / i
                ex_pred = prediction[0].cpu().detach().numpy().round(3)
                ex_target = urt[0].cpu().detach().numpy().round(3)
                train_pbar.set_description("train-avg-loss::{}  "
                                           "--------  rmse-loss::{}  "
                                           "--------  mae-loss::{}  "
                                           "--------  prediction::{}  "
                                           "--------  target::{}  "
                                           .format(avg_loss, loss_, mae_loss, ex_pred,
                                                   ex_target))
                self.train_writer.add_scalar('train__loss', loss_, train_step)
                if train_step % 1000 == 0:
                    self.train_writer.add_text('train_prediction', str(ex_pred), train_step)
                    self.train_writer.add_text('train_target', str(ex_target), train_step)

            # VALIDATION
            tot_loss = 0
            model.eval()
            for i, (xyz, prot_ids, features, urt) in enumerate(val_pbar, start=1):
                val_step += self.batch_size
                xyz = xyz.to(self.device)
                featZ = prot_ids.view(xyz.shape[0], xyz.shape[1]).to(self.device)
                if self.type == 1:
                    prediction = model(xyz, featZ, None)
                elif self.type == 2:
                    feat23 = features.to(self.device)
                    prediction = model(xyz, None, feat23)
                elif self.type == 3:
                    feat23 = features.to(self.device)
                    prediction = model(xyz, featZ, feat23)
                target = urt.to(self.device).double()
                loss = criterion(prediction, target)
                mae = mae_criterion(prediction, target)
                loss_ = torch.sqrt(loss).cpu().item()
                mae_loss = mae.cpu().item()
                tot_loss += mae_loss
                avg_loss = tot_loss / i
                ex_pred = prediction[0].cpu().detach().numpy().round(3)
                ex_target = urt[0].cpu().detach().numpy().round(3)
                val_pbar.set_description("val-avg-loss::{}  "
                                         "--------  rmse-loss::{}  "
                                         "--------  mae-loss::{}  "
                                         "--------  prediction::{}  "
                                         "--------  target::{}  "
                                         .format(avg_loss, loss_, mae_loss, ex_pred,
                                                 ex_target))
                self.val_writer.add_scalar('val__loss', loss_, val_step)
                if val_step % 1000 == 0:
                    self.val_writer.add_text('val_prediction', str(ex_pred), val_step)
                    self.val_writer.add_text('val_target', str(ex_target), val_step)
            if tot_loss < min_loss:
                torch.save(model.state_dict(), self.save_path + '.pkl')
                min_loss = tot_loss
            if epoch > 10:
                scheduler.step()

if __name__ == '__main__':
    # run: module load python_gpu/3.7.1 gcc/6.3.0
    # conda install -c psi4 gcc-5
    # https://scicomp.ethz.ch/wiki/Using_the_batch_system
    # conda env export | grep -v "^prefix: " > environment.yml
    # conda env create -f environment.yml
    parser = argparse.ArgumentParser(description='Specify setting (generates all corresponding .ini files).')
    parser.add_argument('--run', type=int, default=109)
    args = parser.parse_args()
    net = ACN(run_id=args.run)
    net.train_molecular_model()
