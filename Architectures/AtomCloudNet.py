from Architectures.atomcloud import *
from Architectures.cloud_utils import *

import torch
import torch.nn.functional as F
import torch.nn as nn
from functools import partial

import se3cnn
from se3cnn import SO3
if torch.cuda.is_available():
    from se3cnn import real_spherical_harmonics
from se3cnn.point.radial import CosineBasisModel
import se3cnn.non_linearities as nl
from se3cnn.non_linearities import rescaled_act
from se3cnn.point.kernel import Kernel
from se3cnn.point.operations import Convolution, NeighborsConvolution


class se3ACN(nn.Module):
    """
    THIS IS THE ATOMCLOUDNET MODEL

    Modell using xyz (positions of atoms in molecules), Z (atom-ids) and engineered features (2-&3-body-interactions) as
    an input, to abstract
    """
    def __init__(self, device='cpu', nclouds=1, natoms=30, resblocks=1, cloud_dim=24, neighborradius=2,
                 nffl=1, ffl1size=128, emb_dim=32, cloudord=3, nradial=3, nbasis = 10, two_three=False, Z=False):
        super(se3ACN, self).__init__()
        self.device = device
        self.natoms = natoms
        self.two_three = two_three
        self.Z = Z

        self.emb_dim = emb_dim
        if resblocks >= 1:
            self.final_res = True
        else:
            self.final_res = False
        self.cloud_res = True

        self.resblocks = resblocks
        self.cloudnorm = False
        self.feature_collation = 'pool'  # pool or 'sum'
        self.nffl = nffl
        self.ffl1size = ffl1size

        # Cloud specifications
        self.nclouds = nclouds
        self.cloud_order = cloudord
        self.cloud_dim = cloud_dim

        self.radial_layers = nradial
        self.sp = rescaled_act.Softplus(beta=5)
        self.sh = se3cnn.SO3.spherical_harmonics_xyz

        # Embedding
        self.emb = nn.Embedding(num_embeddings=6, embedding_dim=self.emb_dim)

        # Radial Model
        self.number_of_basis = nbasis
        self.neighbor_radius = neighborradius
        self.RadialModel = partial(CosineBasisModel,
                                   max_radius=self.neighbor_radius,
                                   number_of_basis=self.number_of_basis,
                                   h=150,
                                   L=self.radial_layers,
                                   act=self.sp)

        self.K = partial(se3cnn.point.kernel.Kernel, RadialModel=self.RadialModel, sh=self.sh, normalization='norm')

        # Cloud layers
        self.clouds = nn.ModuleList()
        dim_in = self.emb_dim
        # Number output features per atom (these are then stacked with cloud residuals depending on settings...)
        dim_out = self.cloud_dim
        Rs_in = [(dim_in, o) for o in range(1)]
        Rs_out = [(dim_out, o) for o in range(self.cloud_order)]

        for c in range(self.nclouds):
            # Cloud
            self.clouds.append(NeighborsConvolution(self.K, Rs_in, Rs_out, self.neighbor_radius))
            Rs_in = Rs_out

        if self.cloud_res:
            cloud_out = self.cloud_dim * (self.cloud_order ** 2) * self.nclouds
        else:
            cloud_out = self.cloud_dim * (self.cloud_order ** 2)

        # Cloud residuals (should only be applied to final cloud)
        if self.final_res:
            cloud_out += 3  # we add 2 & 3-body interactions
            self.cloud_residual = AtomResiduals(in_channel=cloud_out, res_blocks=self.resblocks, device=self.device)
            res_out = cloud_out * 2
        else:
            res_out = cloud_out

        # passing molecular features through output layer
        self.collate = nn.ModuleList()
        in_shape = res_out
        for _ in range(self.nffl):
            out_shape = self.ffl1size // (_ + 1)
            self.collate.append(nn.Linear(in_shape, out_shape))
            self.collate.append(nn.BatchNorm1d(out_shape))
            in_shape = out_shape

        # Final output activation layer
        self.outputlayer = nn.Linear(in_shape, 1)
        self.act = nn.Sigmoid()  # y is scaled between 0 and 1

    def forward(self, xyz, Z, body23):
        features_emb = None
        features_23 = None
        if self.Z:
            features_emb = self.emb(Z)
        if self.two_three:
            features_23 = body23
        if features_emb is None:
            features = features_23.float()
        else:
            if features_23 is None:
                features = features_emb.float()
            else:
                features = features_emb.float()
        xyz = xyz.to(torch.float64)
        features = features.to(torch.float64)
        feature_list = []
        for _, op in enumerate(self.clouds):
            features = op(features, geometry=xyz)
            if self.cloud_res:
                feature_list.append(features)
            features = features.to(torch.float64)

        if self.cloud_res:
            if len(feature_list) > 1:
                features = torch.cat(feature_list, dim=2)

        # CONCATENATE FEATURES: Z and 2&3-BODY  -->  FEED THEM INTO RESIDUAL LAYER
        if self.final_res:
            features = torch.cat([features_23.float(), features.float()], dim=2).double()
            features = self.cloud_residual(features)
        if 'sum' in self.feature_collation:
            features = features.sum(1)
        elif 'pool' in self.feature_collation:
            # features = F.adaptive_avg_pool2d(features, (1, features.shape[2]))
            # features = F.lp_pool2d(features, norm_type=1, kernel_size=(features.shape[1], 1), ceil_mode=False)
            features = F.lp_pool2d(features, norm_type=2, kernel_size=(features.shape[1], 1), ceil_mode=False)

        features = features.squeeze()
        for _, op in enumerate(self.collate):
            # features = F.leaky_relu(op(features))
            features = F.softplus(op(features))
        return self.act(self.outputlayer(features))
