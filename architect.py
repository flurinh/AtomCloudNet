from Architectures.atomcloud import *
from Architectures.AtomCloudNet import *
from Architectures.cloud_utils import *
import os
from Processing.loader import *
import glob
from torch.utils.data import DataLoader


path = 'data/hist'


n_batches = 1000
batch_size = 1
real_batch_size = 8
nepochs = 30

feats = ['prot', 'ph']


xyz = torch.randn((n_batches, batch_size, 3, 128))
features = torch.randint(1, 8, (n_batches, batch_size, 128))

posb1 = xyz[0]
featb1 = features[0]
print(featb1)

A = AtomEmbedding(transform=True)




print(A)
t = A(featb1)
print(t.shape)