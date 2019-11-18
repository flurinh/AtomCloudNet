from Architectures.atomcloud import *
from Architectures.AtomCloudNet import *
from Architectures.cloud_utils import *
import os
from Processing.loader import *
import glob
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

path = 'data/hist'


n_batches = 1000
batch_size = 256
real_batch_size = 8
nepochs = 30

feats = ['prot', 'ph']


xyz = torch.randn((n_batches, batch_size, 3, 96))
features = torch.randint(1, 8, (n_batches, batch_size, 96))

posb1 = xyz[0]
featb1 = features[0]


o_O = AtomCloudNet()
print(o_O)


for b in trange(xyz.shape[0]):
    xyz_ = xyz[b]
    features_ = features[b]
    output = o_O(xyz_, features_)
    print(output)