import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


def cloud_sampling(natoms, xyz, features, mode='distance'):
    """
    Return clouds for each atom with n atoms in them. The atoms are ranked according to distance or potential to each
    other and the then each cloud is assigned natoms.
    :param natoms: number of atoms to be selected for each cloud
    :param xyz:
    :param features:
    :param mode:
    :return:
    """
    B, N, C = xyz.shape
    # Todo: Find regime on how to select centroids of interest
    # centroids = get_centroids(xyz, natoms) # [B, npoint, C]
    # new_xyz = index_points(xyz, centroids)
    # idx = query_ball_point(natoms, xyz, new_xyz)
    # grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    # grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if features is not None:
        grouped_points = index_points(features, idx)
        transformed = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        transformed = grouped_xyz_norm
    return xyz, transformed


def dist(atom_coord, xyz):
    """
    Calculate euclidean distance between points.
    :param atom_coord: coordinates of centroid
    :param xyz: coordinates of points in cloud to calulate the distance to
    :return: list of distances between centroid and all points in the cloud
    """
    atom_coord_ = torch.Tensor(xyz.size(), dtype=float).fill(atom_coord)
    return torch.pairwise_distance(atom_coord_, xyz)


def potential(atom_coord, xyz, z_atom, z_cloud):
    """
    Calculate potential function between points
    :param atom_coord:
    :param xyz:
    :param z_atom:
    :param z_cloud:
    :return:
    """
    atom_coord_ = torch.Tensor(xyz.size(), dtype=float).fill(atom_coord)
    z_atom_ = torch.Tensor(z_cloud.size(), dtype=float).fill(z_atom)
    return (z_atom_ * z_cloud) / torch.pairwise_distance(atom_coord_, xyz)