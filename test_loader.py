from Processing.loader2 import *

# data = qm9_loader(limit=100000, path='data/QM9' + '/*.xyz', type=3, init=True)
# data.__save_data__()
data = qm9_loader(limit=100000, path='data/QM9' + '/*.xyz', type=3, init=False)
data.clean_outliers()
data.get_max_23()