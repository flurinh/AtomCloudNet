import glob
from tqdm import trange, tqdm
from scipy.spatial.distance import pdist, squareform
import math
import numpy as np
import itertools
import random

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

from Processing.utils import *

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable


def angle_triangle(p1, p2, p3):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    num = (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1) + (z2 - z1) * (z3 - z1)
    den = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) * \
          math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2 + (z3 - z1) ** 2)
    assert den > 0, "Division by 0 error occured in points {}".format((p1, p2, p3))
    return math.degrees(math.acos(num / den))


class qm9_loader(Dataset):
    def __init__(self,
                 limit=10000,
                 shuffle=True,
                 path='data/QM9/*.xyz',
                 type = 1,
                 scaled = True,
                 init = False,
                 test = False):
        self.limit = limit
        self.data = {}
        self.type = type
        self.partial = True
        self.filename = 'data/pkl/data_'+str(self.limit)+'.pkl'
        self.scaled = scaled
        self.max_two = 1
        self.max_three = 1
        self.non_preprocessed = init
        self.test_setting = test
        if init:
            files = glob.glob(pathname=path)
            if shuffle:
                random.shuffle(files)
            counter = 0
            print("Dataloader processing files... Trying to accumulate {} training points.".format(self.limit))
            natom_range = [8, 30]
            for file_id, file in enumerate(tqdm(files)):
                if counter < self.limit:
                    data = qm9_xyz(file)
                    if data.natoms is None:
                        pass
                    else:
                        natoms = data.natoms
                        coords = data.coords
                        prots = np.asarray(list(map(float, data.prots[0])))
                        prots_ids = np.asarray(list(map(float, data.prots[1])))
                        partial = np.asarray(list(map(float, data.partial)))

                        # How we pad:
                        #
                        # Padding points are "out of reach" (outside the radius of all real atoms) for the neighborhoods
                        # of the molecule!
                        xyz = np.full((natom_range[1], 3), 20, dtype=float)
                        Z = np.zeros((natom_range[1], 1))
                        padded_partial = np.zeros((natom_range[1], 1))
                        padded_prot_ids = np.zeros((natom_range[1], 1))
                        for i in range(xyz.shape[0]):
                            if coords.shape[0] > i:
                                Z[i] = prots[i]
                                padded_partial[i] = partial[i]
                                padded_prot_ids[i] = prots_ids[i]
                                print(prots_ids[i])
                                for j in range(3):
                                    xyz[i, j] = coords[i, j]
                        two = None
                        three = None
                        if self.type > 1:
                            two = self.two_body(xyz, Z)
                            three = self.three_body(xyz, Z)
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
                                     'three': three,
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
                                     'gap': gap,
                                     'elect_spa_ext': elect_spa_ext,
                                     'zeropointvib': zeropointvib,
                                     'u0': u0,  # Internal energy at 0K
                                     'Urt': Urt,  # Internal energy at 298.15K  <========== THIS IS THE TARGET PROPERTY!
                                     'Hrt': Hrt,  # Enthalpy at 298.15K
                                     'Grt': Grt,  # Free energy at 298.15K
                                     'heatcap': heatcap,
                                     'file': file}
                        self.data.update({str(counter): data_dict})
                        counter += 1
        else:
            print("Trying to load data from pickle...", self.filename, ". Total number of samples:", self.limit)
            self.__load_data__()
            self.get_max_23()
        self.clean_outliers()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prots = self.data[str(idx)]['prots_ids']
        Z = self.data[str(idx)]['Z']
        name = self.__getfilename__(idx)
        if not self.non_preprocessed:
            # prots = self.fix_padding(prots, Z)
            pass
            two = self.data[str(idx)]['two'].reshape(30, 1)
            three = self.data[str(idx)]['three'].reshape(30, 1)
            if self.scaled:
                two /= self.max_two
                three /= self.max_three
            stack = np.concatenate([Z, two, three], axis=1)
            if not self.test_setting:
                return torch.Tensor(self.data[str(idx)]['xyz']), \
                       torch.LongTensor(prots), \
                       torch.Tensor(stack), \
                       torch.Tensor([self.data[str(idx)]['Urt']])
            else:
                return torch.Tensor(self.data[str(idx)]['xyz']), \
                       torch.LongTensor(prots), \
                       torch.Tensor(stack), \
                       torch.Tensor([self.data[str(idx)]['Urt']]), \
                       name

    def __save_data__(self):
        if not os.path.isdir('data/pkl'):
            os.mkdir('data/pkl')
        with open(self.filename, 'wb') as fp:
            pickle.dump(self.data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def __load_data__(self):
        try:
            with open(self.filename, 'rb') as fp:
                self.data = pickle.load(fp)
        except IOError as e:
            print('File ' + self.filename + ' not found.')
            print(e.errno)

    def __getfilename__(self, idx):
        return self.data[str(idx)]['file']

    def two_body(self, xyz, Z, norm=False):
        dists = squareform(pdist(xyz, 'euclidean', p=2, w=None, V=None, VI=None))
        dists = dists ** 6
        zz = np.outer(Z, Z)
        out = np.asarray([dists[i, j] / zz[i, j] if zz[i, j] != 0 else 0
                          for i, j in itertools.product(range(30), range(30))])
        mask = np.where(out == 0)
        out[mask] = 1
        if norm:
            for i in range(out.shape[0]):
                sigma = 5
                mu = 50
                out[i] = 1 / (sigma * np.sqrt(mu * 2)) * np.e ** -(out[i] ** 2)
        final = 1 / out
        final[mask] = 0
        final = final.reshape((30, 30))
        return final.sum(1)

    def three_body(self, xyz, Z):
        ids = [x for x in range(xyz.shape[0])]
        res = list(itertools.product(*[ids, ids, ids]))
        dists = squareform(pdist(xyz, 'euclidean', p=2, w=None, V=None, VI=None))
        values = [self.three_body_val(ids_, xyz, Z, dists) if len(ids_) == len(set(ids_)) else 0 for ids_ in res]
        grid = np.zeros((30, 30, 30))
        for value, ids_ in zip(values, res):
            x, y, z = ids_
            grid[x, y, z] = value
        sums = np.zeros((xyz.shape[0]))
        for atom in range(xyz.shape[0]):
            sums[atom] = np.sum(grid[atom, :, :])
        return sums

    def three_body_val(self, ids_, xyz, Z, dists, p=3):
        z1 = Z[ids_[0]]
        z2 = Z[ids_[1]]
        z3 = Z[ids_[2]]
        p1 = xyz[ids_[0]]
        p2 = xyz[ids_[1]]
        p3 = xyz[ids_[2]]
        if 20. in p1:
            return 0
        if 20. in p2:
            return 0
        if 20. in p3:
            return 0
        r12 = dists[ids_[0], ids_[1]]
        r23 = dists[ids_[1], ids_[2]]
        r13 = dists[ids_[0], ids_[2]]
        a = self.angle_triangle(xyz[ids_[0]], xyz[ids_[1]], xyz[ids_[2]])
        b = self.angle_triangle(xyz[ids_[1]], xyz[ids_[2]], xyz[ids_[0]])
        c = 180 - a - b
        z_score = z1 * z2 * z3
        angle_score = 1 + math.cos(a) * math.cos(b) * math.cos(c)
        r_score = (r12 * r23 * r13) ** p
        assert r_score > 0, "distance 0 leads to division by zero error! {}".format((r12, r23, r13))
        return z_score * angle_score / r_score

    def get_max_23(self):
        self.max_two = 39.39892996416823
        self.max_three = 93.227861810588
        print("max 2:", self.max_two)
        print("max 3:", self.max_three)

    def clean_outliers(self):
        new_data = {i: _dict for i, [_, _dict] in enumerate(self.data.items()) if (0 <= _dict['Urt'] <= 1)}
        del self.data
        self.data = new_data
        self.limit = len(self.data)
        print("Number of samples after removing outliers:", self.limit)
        new_dict = {}
        keys = [key for key in self.data.keys()]
        for l in range(self.limit):
            new_dict.update({str(l): self.data[keys[l]]})
        del self.data
        self.data = new_dict

    def fix_padding(self, prot_ids, Z):
        """
        This distinguishes between hydrogens and paddings (in the preprocessed data hydrogen and padding atoms are both
        labelled as 0. Set hydrogens to 1, padding atoms to 0
        """
        prot_ids += 1
        prot_ids[np.where(Z == 0)] = 0
        return prot_ids

