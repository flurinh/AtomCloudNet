from utils import *
import glob

filenames = glob.glob('*.pdb')

for filename in filenames:
    get_shift(filename)
    make_dist_file(filename)


#example = '2L7B'
#get_shift(name=example)
#n_shift, h_shift, i = get_shift(name=example)


#xyz, types, res = load_pdb(name=example)
#xyz = np.asarray(xyz)
#res = np.asarray(res)
#make_dist_file(example, xyz, types, res)