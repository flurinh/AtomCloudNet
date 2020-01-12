
from scipy.spatial.distance import pdist, squareform
import numpy as np
import itertools

from Processing.loader2 import *





path = 'data/QM9Train'

data = qm9_loader(limit = 10000, path = path + '/*.xyz')
print(data.data['0']['two'])
for i in range(1000):
    print(data.data[str(i)]['two'])


