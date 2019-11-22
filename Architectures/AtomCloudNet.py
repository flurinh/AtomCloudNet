from Architectures.atomcloud import *
from Architectures.cloud_utils import *
import torch.nn.functional as F


class AtomCloudNet(nn.Module):
    def __init__(self, layers=[512, 256]):
        super(AtomCloudNet, self).__init__()

        self.emb = AtomEmbedding(embedding_dim=128, transform=True)

        self.cloud1 = Atomcloud(natoms=15, nfeats=128, radius=None, layers=[32, 48, 64], include_self=True,
                                retain_features=False, mode='potential')

        self.atom_res1 = AtomResiduals(in_channel=64, res_blocks=2)

        self.cloud2 = Atomcloud(natoms=15, nfeats=128, radius=None, layers=[128, 256, 512], include_self=True,
                                retain_features=False, mode='potential')

        self.atom_res2 = AtomResiduals(in_channel=512, res_blocks=2)

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





"""


First define architecture of layers with atomcloud then call layers with forward


"""

class AtomCloudFeaturePropagation(nn.Module):
    def __init__(self, layers=[512, 256]):
        super(AtomCloudFeaturePropagation, self).__init__()
        final_features = 128

        self.emb = AtomEmbedding(embedding_dim=128, transform=False)

        self.cloud1 = Atomcloud(natoms=4, nfeats=128, radius=None, layers=[128, 256, 512], include_self=True,
                                retain_features=True, mode='potential')
        # if retain_features is True input to the next layer is nfeats +
        # layers[-1] if False layers[-1]

        self.atom_res1 = AtomResiduals(in_channel=640, res_blocks=4)

        self.cloud2 = Atomcloud(natoms=4, nfeats=1280, radius=None, layers=[1280, 512, final_features], include_self=True,
                                retain_features=False, mode='potential')
        """
        self.atom_res2 = AtomResiduals(in_channel=1024, res_blocks=6)

        self.cloud3 = Atomcloud(natoms=2, nfeats=2048, radius=None, layers=[2048, 512, 1], include_self=True,
                                retain_features=False, mode='potential')
        """

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
