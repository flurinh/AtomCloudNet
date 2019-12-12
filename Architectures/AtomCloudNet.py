from Architectures.atomcloud import *
from Architectures.cloud_utils import *

import se3cnn
from se3cnn import SO3
from se3cnn.point.radial import CosineBasisModel
import se3cnn.non_linearities as nl
from se3cnn.non_linearities import rescaled_act
from se3cnn.point.kernel import Kernel
from se3cnn.point.operations import Convolution, NeighborsConvolution

import torch.nn.functional as F
import torch.nn as nn

from functools import partial


class se3AtomCloudNet(nn.Module):
    def __init__(self, device='cpu'):
        super(se3AtomCloudNet, self).__init__()
        # Define all the necessary stats of the network
        self.device = device
        self.emb_dim = 16
        self.list_Rs = [[(self.emb_dim, 0)], [(self.emb_dim, 0), (self.emb_dim, 1), (self.emb_dim, 2), (self.emb_dim, 3)]]
        self.list_Rs2 = [(512, 0)]
        self.list_Rs3 = [(16, 0), (16, 1), (16, 2), (16, 3)]
        self.cloud_dim = 256
        self.radial_layers = 2
        self.sp = rescaled_act.Softplus(beta=5)
        self.max_radius = 1
        self.number_of_basis = 3
        self.RadialModel = partial(CosineBasisModel,
                                   max_radius=self.max_radius,
                                   number_of_basis=self.number_of_basis,
                                   h=100,
                                   L=self.radial_layers,
                                   act=self.sp)
        self.sh = SO3.spherical_harmonics_xyz
        self.K = partial(se3cnn.point.kernel.Kernel, RadialModel=self.RadialModel, sh=self.sh)
        self.neighbor_radius = 1.4

        # Network layers
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=self.emb_dim)
        self.cloud1 = NeighborsConvolution(self.K, self.list_Rs[0], self.list_Rs[1], self.neighbor_radius)
        self.atom_res1 = AtomResiduals(in_channel=self.cloud_dim, res_blocks=4, device=self.device)
        self.molecule_collation = Convolution(self.K, self.list_Rs2, self.list_Rs2)

    def forward(self, features, xyz):
        assert xyz.size()[:2] == features.size()[:2], "xyz ({}) and feature size ({}) should match"\
            .format(xyz.size(), features.size())
        print("0", features.size())
        features = self.emb(features)
        print("1", features.size())
        features = self.cloud1(features, xyz)
        print("2", features.size())
        features = self.atom_res1(features)
        print("3", features.size())
        collation = self.molecule_collation(features, xyz)
        print("4", collation.size())
        return features, collation


class AtomCloudNet(nn.Module):
    def __init__(self, layers=[512, 256], device='cpu'):
        super(AtomCloudNet, self).__init__()
        self.device = device

        self.emb = AtomEmbedding(embedding_dim=self.emb_dim, transform=True, device=self.device)

        self.cloud1 = Atomcloud(natoms=15, nfeats=128, radius=None, layers=[32, 48, 64], include_self=True,
                                retain_features=False, mode='potential', device=self.device)

        self.atom_res1 = AtomResiduals(in_channel=64, res_blocks=2, device=self.device)

        self.cloud2 = Atomcloud(natoms=15, nfeats=128, radius=None, layers=[128, 256, 512], include_self=True,
                                retain_features=False, mode='potential', device=self.device)

        self.atom_res2 = AtomResiduals(in_channel=512, res_blocks=2, device=self.device)

        self.fc1 = nn.Linear(1024, layers[0])
        self.bn1 = nn.BatchNorm1d(layers[0])
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.bn2 = nn.BatchNorm1d(layers[1])
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(layers[1], 1)

    def forward(self, xyz, features):
        print(xyz.shape)
        batch_size, _, _ = xyz.size()
        Z = features
        print("Z", Z.shape)
        emb = self.emb(features)
        print("Embedding:", emb.shape)
        f = self.cloud1(xyz, emb, Z)
        print("Cloudlevel 1:", f.shape)
        f = self.atom_res1(f)
        print("Residual level 1:", f.shape)
        f = self.cloud2(xyz, f, Z)
        print("Cloudlevel 2:", f.shape)
        f = self.atom_res2(f)
        print("Residual level 2:", f.shape)
        centroids = get_centroids(xyz, f)
        print("Centroid data:", centroids.shape)
        f = centroids.view(batch_size, -1)
        f = self.drop1(F.relu(self.bn1(self.fc1(f))))
        f = self.drop2(F.relu(self.bn2(self.fc2(f))))
        f = self.fc3(f)
        f = torch.sigmoid(f)
        return f


class AtomCloudFeaturePropagation(nn.Module):
    def __init__(self, layers=[512, 256], device='cpu'):
        super(AtomCloudFeaturePropagation, self).__init__()
        self.device = device
        final_features = 128

        self.emb = AtomEmbedding(embedding_dim=128, transform=False, device=self.device)

        self.cloud1 = Atomcloud(natoms=4, nfeats=128, radius=None, layers=[128, 256, 512], include_self=True,
                                retain_features=True, mode='potential', device=self.device)
        # if retain_features is True input to the next layer is nfeats +
        # layers[-1] if False layers[-1]
        self.atom_res1 = AtomResiduals(in_channel=640, res_blocks=2, device=self.device)
        self.cloud2 = Atomcloud(natoms=4, nfeats=1280, radius=None, layers=[1280, 1280, final_features],
                                include_self=True, retain_features=False, mode='potential', device=self.device)
        self.fl = nn.Linear(final_features, 1)
        self.act = nn.Sigmoid()

    def forward(self, xyz, features):
        Z = features
        batch_size, _, _ = xyz.size()
        emb = self.emb(features)
        f = self.cloud1(xyz, emb, Z)
        f = self.atom_res1(f)
        f = self.cloud2(xyz, f, Z)
        # print(f[0,:2, :6])
        f = F.adaptive_avg_pool2d(f, (1, f.shape[2]))
        f = self.fl(f)
        f = self.act(f).view(f.shape[0], 1)
        return f
