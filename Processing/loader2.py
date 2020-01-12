import glob
from tqdm import trange, tqdm
from scipy.spatial.distance import pdist, squareform
import numpy as np
import itertools
import random

from Processing.utils import *

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable


# lftp sftp://hidberf@login.leonhard.ethz.ch -e "mirror -v -R -P 16 ~/PycharmProjects/Protos/data/QM9 /cluster/home/hidberf/Protos/data/QM9 ; exit"
# bsub -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=4]" python sandbox.py


class qm9_loader(Dataset):
    def __init__(self,
                 feats=None,
                 limit=np.inf,
                 shuffle=True,
                 path='data/QM9/*.xyz'):
        self.data = {}
        self.partial = True
        # Todo: return features list corresponding to "feats"
        files = glob.glob(pathname=path)
        if shuffle:
            random.shuffle(files)
        counter = 0
        print("Dataloader processing files... Trying to accumulate {} training points.".format(limit))
        natom_range = [8, 30]
        for file_id, file in enumerate(tqdm(files)):
            if counter < limit:
                data = qm9_xyz(file)
                if data.natoms == None:
                    pass
                else:
                    natoms = data.natoms
                    coords = data.coords
                    prots = np.asarray(list(map(float, data.prots[0])))
                    prots_ids = np.asarray(list(map(float, data.prots[1])))
                    partial = np.asarray(list(map(float, data.partial)))

                    # the padding points are out of reach for the neighborhoods of the molecule!
                    xyz = np.full((natom_range[1], 3), 20)
                    # xyz = np.random.rand(natom_range[1], 3)
                    Z = np.zeros((natom_range[1], 1))
                    padded_partial = np.zeros((natom_range[1], 1))
                    padded_prot_ids = np.zeros((natom_range[1], 1))
                    for i in range(xyz.shape[0]):
                        if coords.shape[0] > i:
                            Z[i] = prots[i]
                            padded_partial[i] = partial[i]
                            padded_prot_ids[i] = prots_ids[i]
                            for j in range(3):
                                xyz[i, j] = coords[i, j]

                    two = self.two_body(xyz, Z)
                    # three = self.two_body(xyz, Z)

                    atoms = data.atomtypes
                    properties = data.properties[0]
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
                    Urt = (float(properties[13]) + 200) / (-400)
                    Hrt = float(properties[14])
                    Grt = float(properties[15])
                    heatcap = float(properties[16])

                    data_dict = {'natoms': natoms,
                                 'Z': Z,
                                 'two': two,
                                 'prots_ids': padded_prot_ids,
                                 'partial': padded_partial,
                                 'xyz': xyz,
                                 'rotcon1': rotcon1,
                                 'rotcon2': rotcon2,
                                 'rotcon3': rotcon3,
                                 'dipolemom': dipolemom,
                                 'isotropicpol': isotropicpol,
                                 'homo': homo,
                                 'lumo': lumo,
                                 'gap': gap, # Todo: run
                                 'elect_spa_ext': elect_spa_ext,
                                 'zeropointvib': zeropointvib,
                                 'u0': u0,  # Internal energy at 0K
                                 'Urt': Urt,  # Internal energy at 298.15K Todo: Run
                                 'Hrt': Hrt,  # Enthalpy at 298.15K
                                 'Grt': Grt,  # Free energy at 298.15K
                                 'heatcap': heatcap}
                    self.data.update({str(counter): data_dict})
                    counter += 1
            else:
                break

    def two_body(self, xyz, Z, norm=False):
        dists = squareform(pdist(xyz, 'euclidean', p=2, w=None, V=None, VI=None))
        dists = dists ** 6
        zz = np.outer(Z, Z)
        out = np.asarray([dists[i, j] / zz[i, j] if zz[i, j] != 0 else 0
                          for i, j in itertools.product(range(30), range(30))])
        mask = np.where(out == 0)
        out[mask] = 1
        # Todo: Normalization
        for i in range(out.shape[0]):
            if norm:
                sigma = 5
                mu = 50
                out[i] = 1 / (sigma * np.sqrt(mu * 2)) * np.e ** -(out[i] ** 2)
        final = 1 / out
        final[mask] = 0
        final = final.reshape((30, 30))
        return final.sum(1)

    def three_body(self, xyz, Z):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prots = self.data[str(idx)]['prots_ids']
        two = self.data[str(idx)]['two']
        return torch.Tensor(self.data[str(idx)]['xyz']), \
               torch.LongTensor(prots), \
               torch.Tensor(two), \
               torch.Tensor(two), \
               torch.Tensor([self.data[str(idx)]['Urt']])

