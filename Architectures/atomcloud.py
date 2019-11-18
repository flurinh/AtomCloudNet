from Architectures.cloud_utils import *


def create_emb_layer(num_embeddings, embedding_dim):
    return nn.Embedding(num_embeddings, embedding_dim)


class AtomEmbedding(nn.Module):
    def __init__(self, num_embeddings=8, embedding_dim=64, transform=True, transformed_emb_dim=None):
        super(AtomEmbedding, self).__init__()
        """
        Embedding of atoms. Consider Z as a class. Embedding size.
        :param num_embeddings: number of atoms in the input
        :param embedding_dim: number of features generated by the embedding layer
        :param transform: whether to apply a linear layer on the embedding
        :param transformed_emb_dim: number of features generated by the forward layer
        """
        if transformed_emb_dim is None:
            transformed_emb_dim = embedding_dim
        self.transform = transform
        self.emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.transformed_emb = nn.Linear(embedding_dim, transformed_emb_dim)

    def forward(self, Z):
        emb = self.emb_layer(Z)
        transformed_emb = self.transformed_emb(emb)
        if not self.transform:
            return emb
        else:
            return transformed_emb


class AtomResiduals(nn.Module):
    def __init__(self, in_channel, res_blocks):
        r"""
        Calculate Atom Residuals. Output is twice the size of the input
        :param in_channel: number of features of the atom
        :param res_blocks: number of residual blocks
        """
        super(AtomResiduals, self).__init__()
        self.feature_size = in_channel
        self.unnormalized = True
        self.atom_res_blocks = nn.ModuleList()
        self.atom_norm_blocks = nn.ModuleList()
        for _ in range(res_blocks):
            self.atom_res_blocks.append(nn.Linear(in_channel, in_channel))
            self.atom_norm_blocks.append(nn.BatchNorm1d(in_channel))

    def forward(self, features):
        batch_size = features.shape[0]
        features_ = features.permute(1, 0, 2)
        transformed_ = features_
        for atom in range(features.shape[1]):
            input_ = transformed_[atom]
            #print("Input shape", input_.shape)
            for i, res in enumerate(self.atom_res_blocks):
                if batch_size == 1 or self.unnormalized:
                    input_ = F.relu(res(input_))
                else:
                    bn = self.atom_norm_blocks[i]
                    res_features = res(input_)
                    input_ = F.relu(bn(res_features))
                #print("output", input_.shape)
            transformed_[atom] = input_

        new_features = torch.cat([features_, transformed_], axis=2).permute(1, 0, 2)
        #print(new_features.shape)
        return new_features


# Atomclouds should be universal
class AtomcloudVectorization(nn.Module):
    def __init__(self, natoms, nfeats, layers, retain_features, mode):
        r"""
        Atomcloud is the module transforming an atomcloud into a vector - this vector represents the new features of
        the Atomcloud's centroid/center atom. This module takes fixed number of atoms and features input.
        :param natoms: number of atoms to be selected in the cloud
        :param nfeats: number of features per atom
        :param layers: list of <convolution filter size>'s
        """
        super(AtomcloudVectorization, self).__init__()
        self.natoms = natoms
        self.nfeats = nfeats
        self.mode = mode
        self.retain_features = retain_features
        self.apply_vec_transform = True
        self.spatial_abstraction = nn.Linear(nfeats + 3, nfeats)

        self.conv_features = False  # true if convolution filter dimension should be over features instead of atoms
        self.cloud_convs = nn.ModuleList()
        self.cloud_norms = nn.ModuleList()
        if self.conv_features:
            last_channel = nfeats
        else:
            last_channel = natoms
        for out_channel in layers:
            self.cloud_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.cloud_norms.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, features, centroid, cloud):
        # Use a linear layer to learn spatial abstraction
        _, natoms, ncoords = xyz.size()
        _, _, nfeatures = features.size()
        batch_size, cloud_size = cloud.size()

        masked_xyz = torch.zeros((batch_size, cloud_size, ncoords))
        masked_features = torch.zeros((batch_size, cloud_size, nfeatures))

        for b, mask in enumerate(cloud):
            masked_xyz[b] = xyz[b, mask]
            masked_features[b] = features[b, mask]

        if self.apply_vec_transform:
            # print(masked_xyz.shape)
            # print(masked_features.shape)
            masked_features = torch.cat([masked_xyz, masked_features], axis=2)
            #print("Concatenated features:", masked_features.shape)
            masked_features = self.spatial_abstraction(masked_features)
            #print("Spatially abstracted features:", masked_features.shape)

        # Use cloud's feature table as input to convolution, resulting in new features.
        # new_features = masked_features.permute(0, 2, 1)
        new_features = masked_features

        # Todo: Run convolution over cloud representation - This could/should be kernelized -> eg. Gaussian
        for i, conv in enumerate(self.cloud_convs):
            if batch_size == 1:
                new_features = F.relu(conv(new_features))
            else:
                bn = self.cloud_norms[i]
                conv_features = conv(new_features)
                new_features = F.relu(bn(conv_features))
        # Todo: Combining features
        #print("Convolution output shape:", new_features.shape)
        new_features = torch.max(new_features, 2)[0]
        #print("Collapsed output shape:", new_features.shape)
        if self.retain_features:
            new_features = torch.cat([centroid[1], new_features], axis=1)
        #print("Final output shape:", new_features.shape)
        return new_features


class Atomcloud(nn.Module):
    def __init__(self, natoms, nfeats, radius=None, layers=[32, 64, 128], include_self=False, retain_features=False,
                 mode='potential'):
        super(Atomcloud, self).__init__()
        self.natoms = natoms
        self.nfeats = nfeats
        self.include_self = include_self
        self.retain_features = retain_features
        self.out_features = layers[-1]
        self.radius = radius
        self.mode = mode
        self.Z = None
        self.cloud = AtomcloudVectorization(natoms=natoms, nfeats=nfeats, layers=layers,
                                            retain_features=retain_features, mode=mode)

    def forward(self, xyz, features, Z=None):
        xyz = xyz.permute(0, 2, 1)
        xyz_ = xyz
        if features.shape[1] != xyz.shape[1]:
            features = features.permute(0, 2, 1)
        batch_size, natoms, nfeatures = features.size()
        new_features = torch.zeros((batch_size, natoms, self.out_features))
        if self.retain_features:
            new_features = torch.zeros((batch_size, natoms, self.out_features + self.nfeats))
        Z = Z.view(-1, Z.shape[1], 1)
        # Todo: for each atom go through the entire model and generate new features
        # clouds is a list of masks for all atoms
        clouds, dists = cloud_sampling(xyz, Z=Z, natoms=self.natoms, radius=self.radius, mode=self.mode,
                                       include_self=self.include_self)
        for c, cloud in enumerate(clouds):
            centroid = (xyz[:, c], features[:, c])
            # Shift coordinates of xyz to center cloud
            for b in range(batch_size):
                xyz_[b] = xyz[b] - centroid[0][b]
            new_features[:, c] = self.cloud(xyz_, features, centroid, cloud)
        return new_features
