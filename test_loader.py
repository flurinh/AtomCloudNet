from Processing.loader2 import *

data = qm9_loader(limit=5000, path='data/QM9Test' + '/*.xyz', type=3, init=True)
data.__save_data__()
