print("hello")

from Protos import *
"""
xyz, types, res = load_pdb(path=PATH, name='2L7B')
xyz = np.asarray(xyz)
res = np.asarray(res)
make_dist_file(xyz, types, res, name='2L7B', radius=6)"""

limit = np.inf
new_src_path = PATH + 'raw/*.pdb'
filenames = glob.glob(pathname=new_src_path)
print(len(filenames))
if limit < len(filenames):
    filenames = filename[:limit]
print("Starting to preprocess {} proteins...".format(len(filenames)))

id_list = list(name[9:13] for name in filenames)

for i in tqdm(id_list):
    addlineto_xyz(i, mode='hist')
    addlineto_xyz(i, mode='CV')
