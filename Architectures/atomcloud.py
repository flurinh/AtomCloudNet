from Architectures.point_util import *


class AtomResiduals(nn.Module):
    def __init__(self, in_channel, res_blocks):
        """
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


class Atomcloud(nn.Module):
    def __init__(self, natoms, nfeats, layers):
        """
        Atomcloud is the module transforming a atomcloud into a vector - this vector represents the new features of
        the Atomcloud's centroid/center atom. This module takes fixed number of atoms and features input.
        :param in_channel:
        :param mlp:
        """
        super(Atomcloud, self).__init__()
        self.natoms = natoms
        self.cloud_convs = nn.ModuleList()
        self.cloud_norms = nn.ModuleList()
        last_channel = nfeats
        for out_channel in layers:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, features):
        xyz = xyz.permute(0, 2, 1)
        if features is not None:
                features = features.permute(0, 2, 1)
        for i, conv in enumerate(self.cloud_convs):
            bn = self.mlp_bns[i]
            if new_points.shape[0] == 1:
                new_points = F.relu(conv(new_points))
            else:
                new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class AtomcloudCollapse(nn.Module):
    def __init__(self, radius, natoms, in_channel, layers):
        """
        AtomcloudCollapse is the module transforming a atomcloud into a vector - this vector represents the new features of
        the Atomcloud's centroid/center atom. This module selects the corresponding region of interest
        :param radius: radius to be included in the calculation
        :param natoms: number of atoms to
        :param in_channel:
        :param mlp:
        """
        super(AtomcloudCollapse, self).__init__()
        self.radius = radius
        self.natoms = natoms
        self.cloud_convs = nn.ModuleList()
        self.cloud_norms = nn.ModuleList()
        last_channel = in_channel
        for out_channel in layers:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, features, centroid=[0, 0, 0]):
        xyz = xyz.permute(0, 2, 1)
        if features is not None:
            features = features.permute(0, 2, 1)
        new_xyz = xyz
        return new_xyz, features


class AtomcloudVectorization(nn.Module):
    def __init__(self, infeat, outfeat, mode='distance'):
        super(AtomcloudVectorization, self).__init__()
        pass

    def forward(self, xyz, features, centroid):
        avec = None
        return avec