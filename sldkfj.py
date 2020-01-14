from scipy.spatial.distance import pdist, squareform
import numpy as np
import itertools
import math
import torch


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
    ids = [x for x in range(xyz.shape[0])]
    res = list(itertools.product(*[ids, ids, ids]))
    dists = squareform(pdist(xyz, 'euclidean', p=2, w=None, V=None, VI=None))
    values = [three_body_val(ids_, xyz, Z, dists) if len(ids_) == len(set(ids_)) else 0 for ids_ in res]
    grid = np.zeros((30, 30, 30))
    for value, ids_ in zip(values, res):
        x, y, z = ids_
        grid[x, y, z] = value
    sums = np.zeros((xyz.shape[0]))
    for atom in range(xyz.shape[0]):
        sums[atom] = np.sum(grid[atom,:,:])
    print(sums)
    return sums


def three_body_val(ids_, xyz, Z, dists, p = 3):
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
    a = angle_triangle(xyz[ids_[0]], xyz[ids_[1]], xyz[ids_[2]])
    b = angle_triangle(xyz[ids_[1]], xyz[ids_[2]], xyz[ids_[0]])
    c = 180-a-b
    z_score = z1 * z2 * z3
    angle_score = 1 + math.cos(a) * math.cos(b) * math.cos(c)
    r_score = (r12 * r23 * r13) ** p
    assert r_score > 0, print(r12, r23, r13)
    return z_score * angle_score / r_score


def angle_triangle(p1, p2, p3):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    num = (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1) + (z2 - z1) * (z3 - z1)
    den = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) * \
          math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2 + (z3 - z1) ** 2)
    assert den > 0, print("Nenner 0 error: ", p1, p2, p3)
    return math.degrees(math.acos(num / den))

print(torch.utils.data.get_worker_info())
path = 'data/QM9Train'

data = qm9_loader(limit = 100, path = path + '/*.xyz', shuffle=False)

loader = torch.utils.data.DataLoader(data, batch_size=8, num_workers=4)

for sample in enumerate(loader):
    print(sample)




