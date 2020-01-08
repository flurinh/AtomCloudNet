from Architectures.atomcloud import *
from Architectures.cloud_utils import *

import se3cnn
from se3cnn.point.radial import CosineBasisModel
import se3cnn.non_linearities as nl
from se3cnn.non_linearities import rescaled_act
from se3cnn.non_linearities.rescaled_act import relu, sigmoid
from se3cnn.point.kernel import Kernel
from se3cnn.non_linearities import GatedBlock
from se3cnn.point.operations import Convolution, NeighborsConvolution
from se3cnn.point.operations import PeriodicConvolution
import torch.nn.functional as F
import torch.nn as nn
from functools import partial


class se3AtomCloudNet(nn.Module):
    def __init__(self, device='cpu', nclouds = 1, natoms = 30):
        super(se3AtomCloudNet, self).__init__()
        # Define all the necessary stats of the network
        self.device = device
        self.natoms = natoms

        self.emb_dim = 32
        self.residuals = True
        self.resblocks = 1
        self.cloudnorm = True
        self.feature_collation = 'avg' # pool or else use dense layer
        self.nffl = 1
        self.ffl1size = 128

        # Cloud specifications
        self.nclouds = nclouds
        self.cloud_order = 3
        self.cloud_dim = 24

        self.radial_layers = 2
        self.sp = rescaled_act.Softplus(beta=1)
        self.sh = se3cnn.SO3.spherical_harmonics_xyz

        # Embedding
        self.emb = nn.Embedding(num_embeddings=6, embedding_dim=self.emb_dim)

        # Radial Model
        self.number_of_basis = 3
        self.neighbor_radius = 2
        self.RadialModel = partial(CosineBasisModel,
                                   max_radius=self.neighbor_radius,
                                   number_of_basis=self.number_of_basis,
                                   h=100,
                                   L=self.radial_layers,
                                   act=self.sp)
        self.K = partial(se3cnn.point.kernel.Kernel, RadialModel=self.RadialModel, sh=self.sh, normalization='norm')
        self.NC = partial(NeighborsConvolution, self.K, self.neighbor_radius)

        # Cloud layers
        self.clouds = nn.ModuleList()
        cloud_dim = self.emb_dim
        dim_in = self.emb_dim
        dim_out = self.cloud_dim

        for c in range(self.nclouds):
            # Cloud
            Rs_in = [(dim_in, o) for o in range(1)]
            Rs_out = [(dim_out, o) for o in range(self.cloud_order)]
            print("Rs_in", Rs_in)
            print("Rs_out", Rs_out)
            """
            dimensionalities = [2 * L + 1 for mult, L in Rs_out for _ in range(mult)]
            print(dimensionalities)
            norm_activation = nl.norm_activation.NormActivation(dimensionalities,
                                                                rescaled_act.sigmoid,
                                                                rescaled_act.sigmoid)
            print(norm_activation)
            self.clouds.append(nl.gated_block.GatedBlock(Rs_in, Rs_out, self.sp, rescaled_act.sigmoid, Operation=self.NC))
            """
            self.clouds.append(NeighborsConvolution(self.K, Rs_in, Rs_out, self.neighbor_radius))
            """
            self.clouds.append(GatedBlock(Rs_in, Rs_out, 
                                          scalar_activation=self.sp,
                                          gate_activation=rescaled_act.sigmoid,
                                          Operation=self.NC))
            """
            cloud_out = self.cloud_dim * (self.cloud_order**2)

            # Cloud residuals
            if self.residuals:
                self.clouds.append(AtomResiduals(in_channel=cloud_out, res_blocks=self.resblocks, device=self.device))
                res_out = 2 * cloud_out
            dim_in = res_out

        # molecular feature collation (either dense layer or average pooling of atoms)
        self.collate = nn.Linear(self.natoms, 1)
        # passing molecular features through output layer
        self.collate2 = nn.ModuleList()
        in_shape = res_out
        for _ in range(self.nffl):
            out_shape = self.ffl1size // (_ + 1)
            self.collate2.append(nn.Linear(in_shape, out_shape))
            # self.collate2.append(nn.Dropout(.1))
            self.collate2.append(nn.BatchNorm1d(out_shape))
            in_shape = out_shape
        self.outputlayer = nn.Linear(in_shape, 1)
        # output activation layer
        self.act = nn.Sigmoid()


    def forward(self, xyz, features):
        assert xyz.size()[:1] == features.size()[:1], "xyz ({}) and feature size ({}) should match"\
            .format(xyz.size(), features.size())
        #print("0", features.size())
        features = self.emb(features)
        #print("1", features.size())
        for _, op in enumerate(self.clouds):
            features = op(features, geometry=xyz)
            #print(str(i+2), str(features.size()))
        if 'avg' in self.feature_collation:
            features = features.mean(1)
        elif 'pool' in self.feature_collation: # not tested
            features = F.adaptive_avg_pool2d(features, (1, features.shape[2]))
        else:
            features = features.permute(0, 2, 1)
            features = self.collate(features)
        features = features.squeeze()
        for _, op in enumerate(self.collate2):
            features = F.relu(op(features))
        output = self.act(self.outputlayer(features))
        return output


class AtomCloudNet(nn.Module):
    def __init__(self, layers=[512, 256], device='cpu'):
        super(AtomCloudNet, self).__init__()
        self.device = device

        self.emb_dim = 16
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
        f = F.softmax(f, dim=1)
        return f


class AtomCloudFeaturePropagation(nn.Module):
    def __init__(self, layers=[512, 256], device='cpu'):
        super(AtomCloudFeaturePropagation, self).__init__()
        self.device = device
        self.final_features_ = 128
        self.emb = AtomEmbedding(embedding_dim=128, transform=False, device=self.device)
        self.cloud1 = Atomcloud(natoms=4, nfeats=128, radius=None, layers=[128, 256, 512], include_self=True,
                                retain_features=True, mode='potential', device=self.device)
        # if retain_features is True input to the next layer has dim nfeats +
        # layers[-1] else layers[-1]
        self.atom_res1 = AtomResiduals(in_channel=640, res_blocks=2, device=self.device)
        self.cloud2 = Atomcloud(natoms=4, nfeats=1280, radius=None, layers=[1280, 1280, self.final_features_],
                                include_self=True, retain_features=False, mode='potential', device=self.device)

        # This is used to collapse the features space of the atoms to 1 feature
        self.fl = nn.Linear(self.final_features_, 1)
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
