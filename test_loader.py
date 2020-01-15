from Processing.loader2 import *

data = qm9_loader(limit=100000, path='data/QM9' + '/*.xyz', type=3, init=True)
data.__save_data__()