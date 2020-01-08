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


class ACN:
    def __init__(self,
                 run_id=1):
        path = 'config_ACN/config_'
        self.hyperparams = get_config2(run_id=run_id, path=path)
        self.type = self.hyperparams[0]
        print(self.hyperparams)
        self.run_id = run_id
        self.verbose = 1

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

        self.path = 'data/QM9'
        self.batch_size = self.hyperparams[3]
        self.ngpus = 0
        feats = ['prot', 'ph']
        data = qm9_loader(feats=feats, limit=12800, path=self.path + '/*.xyz')
        print("\nTotal number of samples assembled:", data.__len__())
        self.loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)

        # gpu
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if self.device is 'cuda':
            torch.cuda.synchronize()

        self.nepochs = self.hyperparams[2]

        print("Using device:", self.device)

    def train_molecular_model(self):
        history = []
        model = se3AtomCloudNet(device=self.device).float()
        if self.verbose > 0:
            print(model)
        model.train()
        criterion = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=self.hyperparams[1])
        opt.zero_grad()
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        if self.verbose > 0:
            print(model)
            print("Number trainable parameters:", params)
        """
        if torch.cuda.device_count() > 1:
            self.ngpus = torch.cuda.device_count()
            print("Let's use" + str(self.ngpus) + "GPUs!")
            model = torch.nn.DataParallel(model)
        """
        model.to(self.device)
        step = 0
        torch.autograd.set_detect_anomaly(True)
        pbar = tqdm(self.loader)
        for _ in trange(self.nepochs):
            tot_loss = 0
            for i, (xyz, Z, prot_ids, partial, urt) in enumerate(pbar, start=1):
                step += 1
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
                pbar.set_description("epoch-avg-loss::{}  "
                                     "--------  loss::{}  "
                                     "--------  prediction::{}  "
                                     "--------  target::{}  "
                                     .format(avg_loss, loss_, ex_pred,
                                             ex_target))
                self.train_writer.add_scalar('train__loss', loss_, step)
                if step % 10 == 0:
                    self.train_writer.add_text('train_prediction', str(ex_pred), step)
                    self.train_writer.add_text('train_target', str(ex_target), step)
                history.append(loss_)
            # Todo: Implement a validation setting
            # torch.save(model, self.save_path + '.pt')
        return model, history

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
        pbar = tqdm(self.loader)
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
                pbar.set_description("epoch-avg-loss::{}  "
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
    parser = argparse.ArgumentParser(description='Specify setting (generates all corresponding .ini files).')
    parser.add_argument('--run', type=int, default=1)
    args = parser.parse_args()
    net = ACN(run_id=args.run)
    if net.type == 'ACN':
        model, history = net.train_molecular_model()
