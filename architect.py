from Architectures.atomcloud import *
from Architectures.AtomCloudNet import *
from Architectures.cloud_utils import *
import os
from Processing.loader import *
import glob
from torch.utils.data import DataLoader


path = 'data/hist'


n_batches = 1000
batch_size = 28
real_batch_size = 8
nepochs = 30

feats = ['prot', 'ph']


xyz = torch.randn((n_batches, batch_size, 3, 33))
features = torch.randint(1, 8, (n_batches, batch_size, 33))

posb1 = xyz[0]
featb1 = features[0]


A = AtomEmbedding(embedding_dim=64, transform=True)

B = Atomcloud(natoms=16, nfeats=64, radius=None, layers=[64, 128, 128], include_self=True, retain_features=False,
              mode='potential')

o_O = AtomCloudNet()

print(o_O)

out = o_O(xyz[0], features[0])



print(B)
t = A(featb1)
print(t.shape)
t2 = B(posb1, t, featb1)