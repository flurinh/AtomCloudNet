from Architectures.atomcloud import *
from Architectures.cloud_utils import *


class AtomCloudNet(nn.Module):
    def __init__(self, layers = [512, 256]):
        super(AtomCloudNet, self).__init__()

        self.emb = AtomEmbedding(embedding_dim=64, transform=True)

        self.cloud1 = Atomcloud(natoms=16, nfeats=64, radius=None, layers=[64, 96, 128], include_self=True,
                                retain_features=False, mode='potential')

        self.atom_res1 = AtomResiduals(in_channel=128, res_blocks=2)

        self.cloud2 = Atomcloud(natoms=16, nfeats=64, radius=None, layers=[256, 384, 512], include_self=True,
                                retain_features=False, mode='potential')

        self.atom_res2 = AtomResiduals(in_channel=384, res_blocks=2)

        self.fc1 = nn.Linear(768, layers[0])
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
        f1 = self.cloud1(xyz, emb, Z)
        print("Cloudlevel 1:", f1.shape)
        f1 = self.atom_res1(f1)
        print("Residual level 1:", f1.shape)
        f2 = self.cloud2(xyz, f1, Z)
        print("Cloudlevel 2:", f2.shape)
        f2 = self.atom_res2(f2)
        print("Residual level 2:", f2.shape)

        centroids = get_centroids(f2)

        f = centroids.view(batch_size, 1024)
        f = self.drop1(F.relu(self.bn1(self.fc1(f))))
        f = self.drop2(F.relu(self.bn2(self.fc2(f))))
        f = self.fc3(f)
        f = F.sigmoid(f)
        return f
