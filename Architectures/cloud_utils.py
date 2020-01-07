import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


def get_centroids(xyz, features=None):
    #print("Centroid:", xyz[:, 0, 32])
    #print("Centroid:", xyz[:, 1, 32])
    #print("Centroid:", xyz[:, 2, 32])
    ids_x = torch.where(xyz[:, 0, :] == 0)
    ids_y = torch.where(xyz[:, 1, :] == 0)
    ids_z = torch.where(xyz[:, 2, :] == 0)
    return features[ids_x]


def cloud_sampling(xyz, Z, natoms, radius=None, include_self=True, mode='distance'):
    """
    Return clouds for each atom with n atoms in them. The atoms are ranked according to distance or potential to each
    other and the then each cloud is assigned natoms.
    :param xyz:
    :param Z:
    :param natoms: number of atoms to be selected for each cloud
    :param radius: minimum distance from core
    :param mode: 'distance' or 'potential'
    :return: cloud, cloud_dists
    """
    cloud_dists = None  # [B, N]  # distance vectors
    clouds = []  # [B, N, N]  # mask

    if mode is 'potential' and Z is not None:
        dists = inverse_coulomb_dist(xyz, Z)
    else:
        dists = euclidean_dist(xyz)

    # Todo: select the top natoms (closest)
    for a in range(xyz.shape[1]):
        # Todo: given the mode rank all atoms
        clouds.append(cloud_mask(dists[:, a, :], natoms, include_self=include_self))
    return clouds, cloud_dists


def cloud_mask(dists, natoms, include_self):
    """
    Returns a mask for neighborhood given a distance matrix
    :param dists: distances between atoms
    :param natoms: number of atoms to be selected
    :param include_self: whether to include itself
    :return:
    """
    dists_sorted = torch.argsort(dists, dim=1, descending=False)
    if not include_self:
        natoms += 1
    mask = dists_sorted[:, :natoms]
    if include_self:
        return mask
    else:
        return mask[:, 1:]


def euclidean_dist(xyz):
    """
    Calculate euclidean distance between points.
    :param a: centroid ID
    :param xyz: coordinates of points in cloud to calulate the distance to
    :return: list of distances between centroid and all points in the cloud
    """
    batch_size, natoms, _ = xyz.shape
    dist = torch.sum(xyz ** 2, -1).view(batch_size, natoms, 1) + torch.sum(xyz ** 2, -1).view(batch_size, 1, natoms) - \
           2 * torch.matmul(xyz, xyz.permute(0, 2, 1))
    return dist


def inverse_coulomb_dist(xyz, Z):
    """
    Calculate coulomb distance between points
    :param atom_coord:
    :param xyz:
    :param z_atom:
    :param z_cloud:
    :return:
    """
    batch_size, natoms, _ = xyz.shape
    dist = torch.sum(xyz ** 2, -1).view(batch_size, natoms, 1) + torch.sum(xyz ** 2, -1).view(batch_size, 1, natoms) - \
           2 * torch.matmul(xyz, xyz.permute(0, 2, 1))
    dist = torch.pow(dist, 3).float()
    qq = torch.matmul(Z, Z.permute(0, 2, 1)).float()
    return dist / qq


def electrostatic_dist(xyz, features):
    # q * q / r
    return None
