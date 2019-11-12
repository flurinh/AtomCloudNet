import glob
from tqdm import trange, tqdm
import numpy as np
import pandas as pd

from Processing.utils import *

import torch
from torch.utils.data import Dataset


class xyz_loader(Dataset):
    def __init__(self,
                 limit=np.inf,
                 path='data/hist/*.xyz'):
        self.data = {}
        max_length = 120
        files = glob.glob(pathname=path)
        counter = 0
        print("Dataloader processing files...")

        # TODO: only add valid proteins
        rt_range = []
        salconc_range = []
        ph_range = []

        for file_id, file in enumerate(tqdm(files)):
            if file_id < limit:
                data = load_xyz(file)
                if data.title is not None:
                    data_ = data.title
                    natoms = data_[0]
                    ph = data_[1]
                    T = data_[2]
                    salconc = data_[3]
                    name = data_[4]
                    n_shift = data_[5]
                    h_shift = data_[6]
                    coords = data.coords
                    xyz = np.zeros((coords.shape[0], 3))
                    for i in range(coords.shape[0]):
                        for j in range(coords.shape[1]):
                            xyz[i, j] = coords[i, j]
                    atoms = data.atomtypes
                    data_dict = {'name': name,
                                 'natoms': natoms,
                                 'ph': ph,
                                 'salconc': salconc,
                                 'T': T,
                                 'xyz': xyz,
                                 'prots': data.prots,
                                 '15N-shift': n_shift,
                                 '1H-shift': h_shift}

                    self.data.update({str(counter): data_dict})
                    counter += 1
                else:
                    pass
            else:
                break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[str(idx)]['xyz'], dtype=float), \
               torch.tensor(self.data[str(idx)]['prots'], dtype=float)



class euclid_tensor(Dataset):
    def __init__(self,
                 limit,
                 path='data/hist/*.xyz'):
        self.data = {}
        bins = 100
        max_length = 120
        files = glob.glob(pathname=path)
        counter = 0
        print("Dataloader processing files...")
        for file_id, file in enumerate(tqdm(files)):
            if file_id < limit:
                data = load_xyz(file)
                if data.title is not None:
                    data_ = data.title.split(' ')
                    name = data_[0]
                    xyz = np.zeros((max_length, 3))
                    n_shift = data_[1]
                    h_shift = data_[2]
                    coords = data.coords
                    for i in range(min(coords.shape[0], max_length)):
                        for j in range(coords.shape[1]):
                            xyz[i, j] = coords[i, j]
                    atoms = data.atomtypes
                    data_dict = {'name': name,
                                 'x': xyz[:, 0],
                                 'y': xyz[:, 1],
                                 'z': xyz[:, 2],
                                 'atoms': atoms,
                                 '15N-shift': n_shift,
                                 '1H-shift': h_shift}
                    self.data.update({str(counter): data_dict})
                    counter += 1
                else:
                    pass
            else:
                break
        self.data = pd.DataFrame.from_dict(self.data).T
        self.euclid_x = []
        for i in range(len(self.data)):
            self.euclid_x.append(pd.cut(self.data['x'][0], bins=bins))

        print("euclid:", self.euclid_x[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[str(idx)]['x'], self.data[str(idx)]['atoms']
