# from Processing.download_pdb import *
from Processing.utils import *

from tqdm import tqdm, trange
import argparse
import os

PATH = 'data/'

preprocess = True
limit = 5


class Protos:
    def __init__(self,
                 download = False,
                 preprocess = True,
                 radius = 6,
                 analysis = False):
        self.radius = radius

        if preprocess:
            print("Starting to preprocess.")
            filenames = glob.glob(pathname=PATH+'*_pdb.txt')
            id_list = list(name[5:9] for name in filenames)
            if not os.path.isdir(PATH+'hist'):
                os.mkdir(PATH+'hist')

            for i in id_list:
                xyz, types, res = load_pdb(name=i)
                xyz = np.asarray(xyz)
                res = np.asarray(res)
                make_dist_file(i, xyz, types, res)

            clean_XYZ()
            clean_XYZ()

            os.system('for file in data/hist/*.txt.txt.txt; do mv "$file" "${file/.txt.txt.txt/.xyz}"; done')
            os.system('rm data/hist/*.txt')

            if not os.path.isdir(PATH+'shift'):
                os.mkdir(PATH+'shift')
            if not os.path.isdir(PATH+'hist_noshift'):
                os.mkdir(PATH+'hist_noshift')

            for i in id_list:
                get_shift(i)
            mv_Res_without_Shift()

            for i in id_list:
                addlineto_xyz(i)
            strip_txt_file()

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify run (respective .ini file holds all tunable hyperparameters.')
    parser.add_argument('--radius', type=int, help='Please specify radius.')
    # parser.add_argument('--run', type=int, help='Please specify run ID.')
    # parser.add_argument('--verbose', type=int, default=0, help='Please specify verbose.')
    args = parser.parse_args()
    p = Protos()