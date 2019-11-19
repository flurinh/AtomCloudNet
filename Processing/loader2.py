import glob
from tqdm import trange, tqdm
import numpy as np
import pandas as pd
import random

from Processing.utils import *

import torch
from torch.utils.data import Dataset

from torch.autograd import Variable


class qm9_loader(Dataset):
    def __init__(self,
                 feats,
                 limit=np.inf,
                 shuffle=True,
                 path='data/QM9/*.xyz'):
        self.data = {}
        self.partial = False

        files = glob.glob(pathname=path)
        if shuffle:
            random.shuffle(files)
        counter = 0
        print("Dataloader processing files... Trying to accumulate {} training points.".format(limit))
        # TODO: only add valid proteins

        natom_range = [4, 30]
        max_len = natom_range[1]
        for file_id, file in enumerate(tqdm(files)):
            if counter < limit:
                data = qm9_xyz(file)
                if data.natoms == None:
                    pass
                else:
                    natoms = data.natoms
                    coords = data.coords
                    xyz = np.zeros((coords.shape[0], 3))
                    for i in range(coords.shape[0]):
                        for j in range(coords.shape[1]):
                            xyz[i, j] = coords[i, j]
                    atoms = data.atomtypes
                    prots = data.prots
                    partial = np.asarray(list(map(float, data.partial)))
                    #print(partial)
                    properties = data.properties[0]
                    #print(properties)
                    rotcon1 = float(properties[2])
                    rotcon2 = float(properties[3])
                    rotcon3 = float(properties[4])
                    dipolemom = float(properties[5])
                    isotropicpol = float(properties[6])
                    homo = float(properties[7])
                    lumo = float(properties[8])
                    gap = float(properties[9])
                    elect_spa_ext = float(properties[10])
                    zeropointvib = float(properties[11])
                    u0 = float(properties[12])
                    Urt = float(properties[13])
                    Hrt = float(properties[14])
                    Grt = float(properties[15])
                    heatcap = float(properties[16])

                    data_dict = {'natoms': natoms,
                                 'prots': prots,
                                 'partial': partial,
                                 'xyz': coords,
                                 'rotcon1': rotcon1,
                                 'rotcon2': rotcon2,
                                 'rotcon3': rotcon3,
                                 'dipolemom': dipolemom,
                                 'isotropicpol': isotropicpol,
                                 'homo': homo,
                                 'lumo': lumo,
                                 'gap': gap,
                                 'elect_spa_ext': elect_spa_ext,
                                 'zeropointvib': zeropointvib,
                                 'u0': u0,
                                 'Urt': Urt,
                                 'Hrt': Hrt,
                                 'Grt': Grt,
                                 'heatcap': heatcap}
                    self.data.update({str(counter): data_dict})
                    counter += 1
            else:
                break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p = np.asarray(self.data[str(idx)]['prots'])
        if self.partial:
            feat = np.vstack((p.astype(float), np.full(p.shape, self.data[str(idx)]['partial'])))
        else:
            feat = p
            print(p)
        return torch.Tensor(self.data[str(idx)]['xyz']), torch.LongTensor(feat), \
               torch.Tensor(self.data[str(idx)]['partial'])
