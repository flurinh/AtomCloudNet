import glob
from tqdm import trange, tqdm
import numpy as np
import pandas as pd
import random

from Processing.utils import *

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from torch.autograd import Variabl


class xyz_loader(Dataset):
    def __init__(self,
                 feats,
                 limit=np.inf,
                 shuffle=True,
                 path='data/hist/*.xyz'):
        self.data = {}
        self.ph = False
        if 'ph' in feats:
            self.ph = True

        max_length = 120
        files = glob.glob(pathname=path)
        if shuffle:
            random.shuffle(files)
        counter = 0
        print("Dataloader processing files... Trying to accumulate {} training points.".format(limit))

        # TODO: only add valid proteins
        rt_range = []
        salconc_range = []
        self.ph_list = []
        ph_range = [4.5, 8]
        natom_range = [40, 150]
        for file_id, file in enumerate(tqdm(files)):
            if counter < limit:
                data = load_xyz(file)
                if data.title is not None:
                    data_ = data.title
                    natoms = int(data_[0])
                    ph = data_[1]
                    T = data_[2]
                    salconc = data_[3]
                    name = data_[4]
                    n_shift = (float(data_[5]) - 100) / 50
                    h_shift = (float(data_[6]) - 5) / 10
                    coords = data.coords
                    xyz = np.zeros((coords.shape[0], 3))
                    for i in range(coords.shape[0]):
                        for j in range(coords.shape[1]):
                            xyz[i, j] = coords[i, j]
                    atoms = data.atomtypes
                    if natom_range[0] < natoms < natom_range[1]:
                        if ph.isdigit():
                            self.ph_list.append(float(ph))
                            if ph_range[0] <= float(ph) <= ph_range[1]:
                                data_dict = {'name': name,
                                             'natoms': natoms,
                                             'ph': float(ph),
                                             'salconc': salconc,
                                             'T': T,
                                             'xyz': xyz,
                                             'prots': data.prots,
                                             '15N-shift': n_shift,
                                             '1H-shift': h_shift}
                                self.data.update({str(counter): data_dict})
                                counter += 1
            else:
                break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p = np.asarray(self.data[str(idx)]['prots'])
        if self.ph:
            feat = np.vstack((p.astype(float), np.full(p.shape, self.data[str(idx)]['ph'])))
        else:
            feat = self.data[str(idx)]['prots']
        return torch.DoubleTensor(self.data[str(idx)]['xyz']), \
               torch.DoubleTensor(feat), \
               torch.tensor(self.data[str(idx)]['15N-shift']), \
               torch.tensor(self.data[str(idx)]['1H-shift'])

    def plot_hist(self):
        bins = np.linspace(3.0, 10.0, 71)
        print(bins)
        plt.hist(self.ph_list, bins = np.linspace(3.0, 10.0, 70))
        plt.show()


class coul_loader(Dataset):
    def __init__(self,
                 limit=np.inf,
                 limit_atoms=120,
                 shuffle=True,
                 shuffle_coul=False,
                 path='data/CV/*.xyz'):
        self.data = {}
        self.ph = False

        files = glob.glob(pathname=path)
        if shuffle:
            random.shuffle(files)
        counter = 0
        print("Dataloader processing files... Trying to accumulate {} training points.".format(limit))

        # TODO: only add valid proteins
        rt_range = []
        salconc_range = []
        self.ph_list = []
        ph_range = [4.5, 8]
        natom_range = [limit_atoms-60, limit_atoms]
        for file_id, file in enumerate(tqdm(files)):
            if counter < limit:
                data = load_xyz_cv(file)
                if data.title is not None:
                    data_ = data.title
                    natoms = int(data_[0])
                    ph = data_[1]
                    T = data_[2]
                    salconc = data_[3]
                    name = data_[4]
                    n_shift = (float(data_[5]) - 100) / 50
                    h_shift = (float(data_[6]) - 5) / 10
                    coulombs = np.asarray(data[1])
                    if natom_range[0] < natoms < natom_range[1]:
                        padded_coul = np.zeros(natom_range[1], dtype=float)
                        for i in range(coulombs.shape[0]):
                            padded_coul[i] = coulombs[i]
                        if shuffle_coul:
                            np.random.shuffle(padded_coul)
                        data_dict = {'name': name,
                                     'natoms': natoms,
                                     'salconc': salconc,
                                     'T': T,
                                     'coulombs': padded_coul,
                                     '15N-shift': n_shift,
                                     '1H-shift': h_shift}
                        self.data.update({str(counter): data_dict})
                        counter += 1
            else:
                break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[str(idx)]['coulombs'], dtype=float), \
               torch.tensor(self.data[str(idx)]['15N-shift'], dtype=float), \
               torch.tensor(self.data[str(idx)]['1H-shift'], dtype=float)