import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from Architectures.point_module import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

# SSG: singlescale grouping
class PointNet2ClsSsg(nn.Module):
    def __init__(self, nfeats):
        super(PointNet2ClsSsg, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=32, radius=1, nsample=8, in_channel=nfeats + 3,
                                          mlp=[256, 512, 1024], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=16, radius=1, nsample=8, in_channel=1024 + 3,
                                          mlp=[256, 512, 1024], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=1024 + 3,
                                          mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, xyz, feat):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, feat)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        """print(" ========================= ")
        print(xyz.shape)
        print(feat.shape)
        print()
        print(l1_xyz.shape)
        print(l1_points.shape)
        print()
        print(l2_xyz.shape)
        print(l2_points.shape)
        print()
        print(l3_xyz.shape)
        print(l3_points.shape)
        print(" ========================= ")
        print()"""
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu((self.fc1(x)))) # i deleted batchnorm
        x = self.drop2(F.relu((self.fc2(x)))) # i deleted batchnorm
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x
