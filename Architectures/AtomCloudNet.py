from Architectures.atomcloud import *
from Architectures.cloud_utils import *


class AtomCloudNet(nn.Module):
    def __init__(self):
        super(AtomCloudNet, self).__init__()
        self.cloud1 = Atomcloud(natoms=16, nfeats=32, layers=[32, 64, 128], mode='distance')
        self.atom_res1 = AtomResiduals(in_channel=33, res_blocks=2)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, xyz, features):
        batch_size, _, _ = xyz.shape
        c1, f1 = self.cloud1(xyz, features)
        c1, f1 = self.atom_res1(xyz, features, c1, f1)
        c2, f2 = self.cloud2(c1, f1)
        c2, f2 = self.atom_res2(xyz, features, c2, f2)
        c3, f3 = self.cloud3(c2, f2)
        c3, f3 = self.atom_res3(xyz, features, c3, f3)

        cloud_vec = self.cloud2vec(c3, f3)

        x = cloud_vec.view(batch_size, 1024)
        x = self.drop1(F.relu((self.fc1(x)))) # i deleted batchnorm
        x = self.drop2(F.relu((self.fc2(x)))) # i deleted batchnorm
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x
