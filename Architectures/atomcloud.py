from Architectures.cloud_utils import *


class AtomResiduals(nn.Module):
    def __init__(self, in_channel, res_blocks):
        r"""
        Calculate Atom Residuals. Output is twice the size of the input
        :param in_channel: number of features of the atom
        :param res_blocks: number of residual blocks
        """
        super(AtomResiduals, self).__init__()
        self.feature_size = in_channel
        self.atom_res_blocks = nn.ModuleList()
        for _ in range(res_blocks):
            self.atom_res_blocks.append(nn.Dense(in_channel, in_channel))
            self.atom_res_blocks.append(nn.ReLU(in_channel))

    def forward(self, features):
        features = features.view(-1, self.feature_size)
        transformed_ = self.atom_res_blocks(features)
        new_features = torch.cat([features, transformed_], axis=1)
        print(new_features.shape)
        return new_features


# Atomclouds should be universal
class AtomcloudVectorization(nn.Module):
    def __init__(self, natoms, nfeats, layers, mode):
        r"""
        Atomcloud is the module transforming an atomcloud into a vector - this vector represents the new features of
        the Atomcloud's centroid/center atom. This module takes fixed number of atoms and features input.
        :param natoms: number of atoms to be selected in the cloud
        :param nfeats: number of features per atom
        :param layers: list of <convolution filter size>'s
        """
        super(AtomcloudVectorization, self).__init__()
        self.natoms = natoms
        self.mode = mode
        self.cloud_convs = nn.ModuleList()
        self.cloud_norms = nn.ModuleList()
        last_channel = nfeats
        for out_channel in layers:
            self.cloud_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.cloud_norms.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, features, centroid):
        # Todo: Find natoms closest atoms to centroid => corresponding cloud
        cloud = cloud_sampling(xyz, features, self.natoms, self.mode)
        # Todo: Use cloud's feature table as input to convolution, resulting in new features.

        # Todo: Run convolution over cloud representation - This could/should be kernelized!

        for i, conv in enumerate(self.cloud_convs):
            bn = self.cloud_norms[i]
            if new_features.shape[0] == 1:
                new_points = F.relu(conv(new_features))
            else:
                new_features = F.relu(bn(conv(new_features)))

        new_features = torch.max(new_features, 2)[0]
        return new_features


class Atomcloud(nn.Module):
    def __init__(self, natoms, nfeat, layers=[32, 64, 128], mode='distance'):
        super(Atomcloud, self).__init__()
        self.cloud = AtomcloudVectorization(natoms=natoms, nfeats=nfeat, layers=layers, mode=mode)

    def forward(self, xyz, features):
        xyz = xyz.permute(0, 2, 1)
        if features is not None:
            features = features.permute(0, 2, 1)
        # Todo: for each atom go through the entire model and generate new features
        for i in range(xyz.shape[0]):
            centroid = (xyz[i], features[i])
            # Shift coordinates of xyz to center cloud
            xyz_ = xyz - centroid[0]
            features[i] = self.cloud(xyz_, features, centroid)
        return xyz, features
