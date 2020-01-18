from Architectures.AtomCloudNet import *
from Processing.loader import *
from utils import *

from collections import OrderedDict
from tqdm import tqdm, trange
from configparser import ConfigParser
import numpy as np
import ast
import json
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
        self.path = 'config_ACN/config_'
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
        if not os.path.isdir('runs/run_{}'.format(int(self.run_id) // 100)):
            os.mkdir('runs/run_{}'.format(int(self.run_id) // 100))
        if not os.path.isdir(self.val_path):
            os.mkdir(self.val_path)
        if not os.path.isdir(self.train_path):
            os.mkdir(self.train_path)
        if not os.path.isdir(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)

    def load_model_specs(self):
        self.hyperparams = get_config2(run_id=self.run_id, path=self.path)
        print("MODEL SPEC", self.hyperparams)
        self.train_path_ = TRAIN_PATH
        self.test_path_ = TEST_PATH
        self.type = self.hyperparams[0]
        self.val_size = .1
        self.val_writer = SummaryWriter(self.val_path)
        self.train_writer = SummaryWriter(self.train_path)
        self.batch_size = self.hyperparams[7]
        self.ngpus = 0
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Using device:", self.device)
        self.use_Z_emb = False
        self.use_23_body = False
        if self.type == 1:
            self.use_Z_emb = True
        if self.type == 2:
            self.use_23_body = True
        if self.type == 3:
            self.use_Z_emb = True
            self.use_23_body = True

    def eval_molecular_model(self):
        """
        Evaluation of the trained model. An evaluation file is created in the models/run_xxx folder is created, which
        can be evaluated with our "results.py" file.
        :return:
        """
        test_data = qm9_loader(limit=5000, path=self.test_path_ + '/*.xyz', type=self.type, init=False, test=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

        # Load model
        with torch_default_dtype(torch.float64):
            print("Loading model from", self.save_path)
            model = se3ACN(device=self.device, nclouds=self.hyperparams[3], natoms=30,
                           resblocks=self.hyperparams[5], cloud_dim=self.hyperparams[4],
                           neighborradius=self.hyperparams[2],
                           nffl=self.hyperparams[8], ffl1size=self.hyperparams[9], emb_dim=self.hyperparams[10],
                           cloudord=self.hyperparams[11], nradial=self.hyperparams[12], nbasis=self.hyperparams[13],
                           two_three=self.use_23_body, Z=self.use_Z_emb).to(self.device)
            state_dict = torch.load(self.save_path + '.pkl', map_location=torch.device(self.device))

            if self.run_id == 14003 or self.run_id == 14006 or self.run_id == 15001 or self.run_id == 15002 or \
                    self.run_id == 15003 or self.run_id == 15006:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)
            model.eval()
            criterion = nn.MSELoss()
            mae_criterion = nn.L1Loss()
            tot_loss = 0
            mae_losses = []
            rmse_losses = []
            predictions = []
            targets = []
            target_names = []

            test_pbar = tqdm(test_loader)
            for i, (xyz, prot_ids, features, urt, names) in enumerate(test_pbar, start=1):
                xyz = xyz.to(self.device)
                featZ = prot_ids.view(xyz.shape[0], xyz.shape[1]).to(self.device)
                feat23 = features.to(self.device)
                prediction = model(xyz, featZ, feat23)
                target = urt.to(self.device).double()
                loss = criterion(prediction, target)
                mae = mae_criterion(prediction, target)
                loss_ = torch.sqrt(loss).cpu().item()
                mae_loss = mae.cpu().item()
                mae_losses.append(mae_loss)
                rmse_losses.append(loss_)
                tot_loss += mae_loss
                avg_loss = tot_loss / i
                predictions.append(prediction.cpu().detach().numpy())
                targets.append(urt.cpu().detach().numpy())
                target_names.append(list(names))
                ex_pred = prediction[0].cpu().detach().numpy().round(3)
                ex_target = urt[0].cpu().detach().numpy().round(3)
                test_pbar.set_description("test-avg-loss::{}  "
                                          "--------  rmse-loss::{}  "
                                          "--------  mae-loss::{}  "
                                          "--------  prediction::{}  "
                                          "--------  target::{}  "
                                          .format(avg_loss, loss_, mae_loss, ex_pred,
                                                  ex_target))
            results = {'rmse_losses': rmse_losses,
                       'mae_losses': mae_losses,
                       'avg_mae': avg_loss,
                       'predictions': predictions,
                       'targets': targets,
                       'names': target_names}
            f = open(self.save_path + '_eval.txt', "wb")
            pickle.dump(results, f)
            f.close()

    def train_molecular_model(self):
        """
        Training settings
        :return:
        """
        train_data = qm9_loader(limit=1000, path=self.train_path_ + '/*.xyz', type=self.type, init=False)
        self.limit = train_data.limit
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
        self.nepochs = self.hyperparams[6]

        with torch_default_dtype(torch.float64):
            print("Generating model of type {}".format(self.type))
            model = se3ACN(device=self.device, nclouds=self.hyperparams[3], natoms=30,
                           resblocks=self.hyperparams[5], cloud_dim=self.hyperparams[4],
                           neighborradius=self.hyperparams[2],
                           nffl=self.hyperparams[8], ffl1size=self.hyperparams[9], emb_dim=self.hyperparams[10],
                           cloudord=self.hyperparams[11], nradial=self.hyperparams[12], nbasis=self.hyperparams[13],
                           two_three=self.use_23_body, Z=self.use_Z_emb)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
        model.to(self.device)
        criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        opt = torch.optim.Adam(model.parameters(), lr=self.hyperparams[1], weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True)
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

            if tot_loss < min_loss:
                torch.save(model.state_dict(), self.save_path + '.pkl')
                min_loss = tot_loss
            if epoch > 2:
                scheduler.step(avg_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify setting.')
    parser.add_argument('--run', type=int, default=14003)
    parser.add_argument('--mode', type=int, default=0)
    args = parser.parse_args()
    net = ACN(run_id=args.run)
    net.load_model_specs()
    if args.mode == 0:
        net.train_molecular_model()
    elif args.mode == 1:
        net.eval_molecular_model()
    elif args.mode == 2:
        net.train_molecular_model()
        net.eval_molecular_model()

