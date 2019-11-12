import glob
from tqdm import trange, tqdm
import numpy as np
import pandas as pd

from Processing.utils import *

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable


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
        natom_range = [40, 150]

        for file_id, file in enumerate(tqdm(files)):
            if file_id < limit:
                data = load_xyz(file)
                if data.title is not None:
                    data_ = data.title
                    natoms = int(data_[0])
                    ph = data_[1]
                    T = data_[2]
                    salconc = data_[3]
                    name = data_[4]
                    n_shift = float(data_[5]) / 150
                    h_shift = float(data_[6]) / 15
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
                    if natom_range[0] < natoms < natom_range[1]:
                        self.data.update({str(counter): data_dict})
                        counter += 1
            else:
                break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.DoubleTensor(self.data[str(idx)]['xyz']), \
               torch.DoubleTensor(self.data[str(idx)]['prots']), \
               torch.tensor(self.data[str(idx)]['15N-shift']), \
               torch.tensor(self.data[str(idx)]['1H-shift'])

