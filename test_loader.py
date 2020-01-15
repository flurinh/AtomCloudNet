from Processing.loader2 import *

data = qm9_loader(limit=10000, path='data/QM9Train' + '/*.xyz', type=3, init=True)
data.__save_data__()