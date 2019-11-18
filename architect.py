from Architectures.atomcloud import *
from Architectures.AtomCloudNet import *
from Architectures.cloud_utils import *
import os
from Processing.loader import *
import glob
from torch.utils.data import DataLoader


path = 'data/hist'


n_batches = 1000
batch_size = 2
real_batch_size = 8
nepochs = 30

feats = ['prot', 'ph']


xyz = torch.randn((n_batches, batch_size, 3, 128))
features = torch.randint(1, 8, (n_batches, batch_size, 128))

posb1 = xyz[0]
featb1 = features[0]
print(featb1)


A = AtomEmbedding(transform=True)
B = Atomcloud(natoms=16, nfeats=32, radius=None, layers=[64, 128, 256], mode='potential')


print(B)
t = A(featb1)
print(t.shape)
t2 = B(posb1, t, featb1)