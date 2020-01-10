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

        feats = ['prot', 'ph']
        train_data = qm9_loader(feats=feats, limit=np.inf, path=self.train_path + '/*.xyz')
        # test_data = qm9_loader(feats=feats, limit=np.inf, path=self.test_path + '/*.xyz')
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

        # gpu
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if self.device is 'cuda':
            torch.cuda.synchronize()
        print("Using device:", self.device)

        self.nepochs = self.hyperparams[6]

    def train_molecular_model(self):
        model = se3AtomCloudNet(device=self.device, nclouds = self.hyperparams[3], natoms = 30,
                                resblocks = self.hyperparams[5], cloud_dim=self.hyperparams[4],
                                neighborradius=self.hyperparams[2]).float()
        model.train()
        criterion = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=self.hyperparams[1])
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        if self.verbose > 0:
            print(model)
            print("Number trainable parameters:", params)
        torch.autograd.set_detect_anomaly(True)
        model.to(self.device)

        train_step = 0
        val_step = 0
        train_pbar = tqdm(self.train_loader)
        val_pbar = tqdm(self.val_loader)

        min_loss = np.inf

        for _ in trange(self.nepochs):
            # TRAINING
            tot_loss = 0
            model.train()
            for i, (xyz, Z, prot_ids, partial, urt) in enumerate(train_pbar, start=1):
                train_step += 1
                xyz = xyz.to(self.device)
                feat = prot_ids.view(xyz.shape[0], xyz.shape[1]).to(self.device)
                prediction = model(xyz, feat)
                loss = criterion(prediction, urt.to(self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_ = torch.sqrt(loss).cpu().item()
                tot_loss += loss_
                avg_loss = tot_loss / i
                ex_pred = prediction[0].cpu().detach().numpy().round(3)
                ex_target = urt[0].cpu().detach().numpy().round(3)
                train_pbar.set_description("epoch-avg-loss::{}  "
                                           "--------  loss::{}  "
                                           "--------  prediction::{}  "
                                           "--------  target::{}  "
                                           .format(avg_loss, loss_, ex_pred,
                                                   ex_target))
                self.train_writer.add_scalar('train__loss', loss_, train_step)
                if train_step % 1000 == 0:
                    self.train_writer.add_text('train_prediction', str(ex_pred), train_step)
                    self.train_writer.add_text('train_target', str(ex_target), train_step)

            # VALIDATION
            tot_loss = 0
            model.eval()
            for i, (xyz, Z, prot_ids, partial, urt) in enumerate(val_pbar, start=1):
                val_step += 1
                xyz = xyz.to(self.device)
                feat = prot_ids.view(xyz.shape[0], xyz.shape[1]).to(self.device)
                prediction = model(xyz, feat)
                loss = criterion(prediction, urt.to(self.device))
                loss_ = torch.sqrt(loss).cpu().item()
                tot_loss += loss_
                avg_loss = tot_loss / i
                ex_pred = prediction[0].cpu().detach().numpy().round(3)
                ex_target = urt[0].cpu().detach().numpy().round(3)
                val_pbar.set_description("training-avg-loss::{}  "
                                         "--------  loss::{}  "
                                         "--------  prediction::{}  "
                                         "--------  target::{}  "
                                         .format(avg_loss, loss_, ex_pred,
                                                 ex_target))
                self.val_writer.add_scalar('val__loss', loss_, val_step)
                if val_step % 1000 == 0:
                    self.val_writer.add_text('val_prediction', str(ex_pred), val_step)
                    self.val_writer.add_text('val_target', str(ex_target), val_step)
            if tot_loss < min_loss:
                torch.save(model.state_dict(), self.save_path + '.pkl')
                min_loss = tot_loss

    def train_atomic_model(self):
        history = []
        model = AtomCloudFeaturePropagation(device=self.device).float()
        model.train()
        criterion = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt.zero_grad()
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        if self.verbose > 0:
            print(model)
            print("Number trainable parameters:", params)

        if torch.cuda.device_count() > 1:
            self.ngpus = torch.cuda.device_count()
            print("Let's use" + str(self.ngpus) + "GPUs!")
            model = torch.nn.DataParallel(model)
        model.to(self.device)

        torch.autograd.set_detect_anomaly(True)
        pbar = tqdm(self.train_loader)
        for _ in trange(self.nepochs):
            tot_loss = 0
            for i, (xyz, Z, prot_ids, partial, urt) in enumerate(pbar, start=1):
                opt.zero_grad()
                xyz = xyz.permute(0, 2, 1).to(self.device)
                feat = prot_ids.view(xyz.shape[0], xyz.shape[2]).to(self.device)
                prediction = model(feat, xyz)
                loss = criterion(prediction, urt.to(self.device))
                loss_ = torch.sqrt(loss).cpu().item()
                tot_loss += loss_
                avg_loss = tot_loss / i
                pbar.set_description("validation-avg-loss::{}  "
                                     "--------  loss::{}  "
                                     "--------  prediction::{}  "
                                     "--------  target::{}  "
                                     .format(avg_loss, loss_, prediction[0].cpu().detach().numpy(), urt[0].cpu().detach().numpy()))
                loss.backward()
                opt.step()
        return model, history


if __name__ == '__main__':
    # run: module load python_gpu/3.7.1 gcc/6.3.0
    # conda install -c psi4 gcc-5
    # https://scicomp.ethz.ch/wiki/Using_the_batch_system
    #conda env export | grep -v "^prefix: " > environment.yml
    #conda env create -f environment.yml
    parser = argparse.ArgumentParser(description='Specify setting (generates all corresponding .ini files).')
    parser.add_argument('--run', type=int, default=109)
    args = parser.parse_args()
    net = ACN(run_id=args.run)
    if net.type == 'ACN':
        net.train_molecular_model()
