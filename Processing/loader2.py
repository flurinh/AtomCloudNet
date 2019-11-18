import glob
from tqdm import trange, tqdm
import numpy as np
import pandas as pd
import random

from Processing.utils import *

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from torch.autograd import Variable


class qm9_loader(Dataset):
    def __init__(self,
                 feats,
                 limit=np.inf,
                 shuffle=True,
                 path='data/QM9/*.xyz'):
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
        self.ph_list = []
        natom_range = [40, 150]
        for file_id, file in enumerate(tqdm(files)):
            if counter < limit:
                data = qm9_xyz(file)
                if data.title is not None:
                    data_ = data.title
                    natoms = int(data.natoms)
                    name = data_[4]
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
                                             'rotcon1' = rotcon1,
                                             'rotcon2' = rotcon2,
                                             'rotcon3' = rotcon3,
                                             'dipolemom' = dipolemom,
                                             'isotropicpol' = isotropicpol,
                                             'homo' = homo,
                                             'lumo' = lumo,
                                             'gap' = gap,
                                             'eclect' = eclect,
                                             'zeropointvib' = zeropointvib,
                                             'u0' = u0,
                                             'Urt' = Urt,
                                             'Hrt' = Hrt,
                                             'Grt' = Grt,
                                             'heatcap' = heatcap}
                                self.data.update({str(counter): data_dict})
                                counter += 1
            else:
                break