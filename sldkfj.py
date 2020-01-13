from scipy.spatial.distance import pdist, squareform
import numpy as np
import itertools
import math

from Processing.loader2 import *


def two_body(xyz, Z, norm=False):
    dists = squareform(pdist(xyz, 'euclidean', p=2, w=None, V=None, VI=None))
    dists = dists ** 6
    zz = np.outer(Z, Z)
    out = np.asarray([dists[i, j] / zz[i, j] if zz[i, j] != 0 else 0
                      for i, j in itertools.product(range(30), range(30))])
    mask = np.where(out == 0)
    out[mask] = 1
    final = 1 / out
    final[mask] = 0
    final = final.reshape((30, 30))
    print(final.sum(1), Z)
    return final.sum(1)


def three_body(xyz, Z):
    dists = squareform(pdist(xyz, 'euclidean', p=2, w=None, V=None, VI=None))
    dists = dists ** 3
    print(dists.shape)
    """zz = np.outer(Z, Z, Z)
    out = np.asarray([dists[i, j] / zz[i, j] if zz[i, j] != 0 else 0
                      for i, j in itertools.product(range(30), range(30))])
    mask = np.where(out == 0)
    out[mask] = 1
    final = 1 / out
    final[mask] = 0
    final = final.reshape((30, 30, 30))"""
    pass





path = 'data/QM9Train'

data = qm9_loader(limit = 100, path = path + '/*.xyz')

for i in range(1):
    sample = data.__getitem__(i)
    xyz = sample[0].numpy()
    Z = sample[1].numpy()
    three_body(xyz, Z)




