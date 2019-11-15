import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


def cloud_sampling(xyz, features, natoms, mode='distance'):
    """
    Return clouds for each atom with n atoms in them. The atoms are ranked according to distance or potential to each
    other and the then each cloud is assigned natoms.
    :param natoms: number of atoms to be selected for each cloud
    :param xyz:
    :param features:
    :param mode: 'distance' or 'potential'
    :return: cloud
    """
    B, N, C = xyz.shape
    cloud_xyz = None
    cloud_dists = None
    cloud = None  # should have shape [B, natoms, C]
    # Todo: given the mode rank all atoms and select the top natoms (closest)

    # Todo: return the cloud (features only!)
    return cloud_xyz, cloud_dists, cloud


def spatial_search(xyz, features, natoms, mode):
    ids = None
    return ids


def euclidean_dist(xyz):
    """
    Calculate euclidean distance between points.
    :param xyz: coordinates of points in cloud to calulate the distance to
    :return: list of distances between centroid and all points in the cloud
    """
    return torch.sqrt(xyz * xyz)


def coulomb_dist(xyz, features):
    """
    Calculate coulomb distance between points
    :param atom_coord:
    :param xyz:
    :param z_atom:
    :param z_cloud:
    :return:
    """
    dists = torch.sqrt(xyz * xyz)
    npdists = dists.numpy()
    mask = npdists.where(npdists == 0)
    xyz_ = xyz[mask]
    features = features[mask]
    id = mask[0][0]
    z_centroid = features[id]
    cdists = z_centroid * features[:][0] / npdists

    return (z_atom_ * z_cloud) / torch.pairwise_distance(atom_coord_, xyz)


def electrostatic_dist(xyz, features):
    # q * q / r
    return None

