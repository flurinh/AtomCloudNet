from Architectures.atomcloud import *
from Architectures.cloud_utils import *


class AtomCloudNet(nn.Module):
    def __init__(self, layers = [512, 256]):
        super(AtomCloudNet, self).__init__()

        self.emb = AtomEmbedding(embedding_dim=32, transform=True)

        self.cloud1 = Atomcloud(natoms=6, nfeats=32, radius=None, layers=[32, 48, 64], include_self=True,
                                retain_features=False, mode='potential')

        self.atom_res1 = AtomResiduals(in_channel=64, res_blocks=2)

        self.cloud2 = Atomcloud(natoms=6, nfeats=128, radius=None, layers=[128, 128, 128], include_self=True,
                                retain_features=False, mode='potential')

        self.atom_res2 = AtomResiduals(in_channel=128, res_blocks=2)

        self.fc1 = nn.Linear(256, layers[0])
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
        centroids = get_centroids(xyz, f2)
        print("Centroid data:", centroids.shape)
        f = centroids.view(batch_size, -1)
        f = self.drop1(F.relu(self.bn1(self.fc1(f))))
        f = self.drop2(F.relu(self.bn2(self.fc2(f))))
        f = self.fc3(f)
        f = torch.sigmoid(f)
        return f


class AtomCloudFeaturePropagation(nn.Module):
    def __init__(self, layers = [512, 256]):
        super(AtomCloudFeaturePropagation, self).__init__()

        self.emb = AtomEmbedding(embedding_dim=128, transform=True)

        self.cloud1 = Atomcloud(natoms=4, nfeats=128, radius=None, layers=[128, 256, 384], include_self=True,
                                retain_features=False, mode='potential')

        self.atom_res1 = AtomResiduals(in_channel=384, res_blocks=2)

        self.cloud2 = Atomcloud(natoms=4, nfeats=768, radius=None, layers=[768, 768, 1024], include_self=True,
                                retain_features=False, mode='potential')

        self.atom_res2 = AtomResiduals(in_channel=1024, res_blocks=2)

        self.cloud3 = Atomcloud(natoms=4, nfeats=2048, radius=None, layers=[1024, 128, 1], include_self=True,
                                retain_features=False, mode='potential')

    def forward(self, xyz, features):
        #print(xyz.shape)
        Z = features
        #print("Z", Z)
        batch_size, _, _ = xyz.size()
        emb = self.emb(features)
        #print("Embedding:", emb.shape)
        f1 = self.cloud1(xyz, emb, Z)
        #print("Cloudlevel 1:", f1.shape)
        f1 = self.atom_res1(f1)
        #print("Residual level 1:", f1.shape)
        f2 = self.cloud2(xyz, f1, Z)
        #print("Cloudlevel 2:", f2.shape)
        f2 = self.atom_res2(f2)
        #print("Residual level 2:", f2.shape)
        f3 = self.cloud3(xyz, f2, Z).view(-1, f2.shape[1])
        #print(f3.shape)
        return f3