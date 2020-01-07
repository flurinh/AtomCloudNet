from Processing.download_pdb import *
from Processing.utils import *
from Processing.loader import *

from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import argparse
import os

PATH = 'data/'


def check_hist(name):
    hist_files = glob.glob(PATH + 'hist/*')
    for h in hist_files:
        if name in h:
            return True
    return False


class Protos:
    def __init__(self,
                 radius,
                 limit,
                 download=False,
                 preprocess=True,
                 analysis=False):
        self.radius = radius

        if download:
            download_proteins()

        limit = limit
        if preprocess:
            print("preprocessing")
            pathname = PATH + 'raw/*_pdb.txt'
            filenames = glob.glob(pathname=pathname)
            for txt in filenames:
                if '_pdb.txt' in txt:
                    new_src = txt.replace('_pdb.txt', '.pdb')
                    os.rename(txt, new_src)
            new_src_path = PATH+'raw/*.pdb'
            filenames = glob.glob(pathname=new_src_path)
            if limit < len(filenames):
                filenames = filenames[:limit]
            print("Starting to preprocess {} proteins...".format(len(filenames)))

            id_list = list(name[9:13] for name in filenames)
            #id_list = ['2L7B', '2L7Q']

            if not os.path.isdir(PATH + 'hist'):
                os.mkdir(PATH + 'hist')

            print("Creating histograms...")
            for _, i in enumerate(tqdm(id_list)):
                if _ > limit:
                    pass
                else:
                    if check_hist(i):
                        pass
                    else:
                        xyz, types, res = load_pdb(path=PATH + 'raw/', name=i)
                        xyz = np.asarray(xyz)
                        res = np.asarray(res)
                        make_dist_file(xyz, types, res, i, radius=radius)
                        make_dist_vector(xyz, types, res, i, radius=radius)

            print("Cleaning txt files...")
            get_all_stupid_atoms()

            hist_files = glob.glob(PATH + 'hist/*.txt')
            print("Casting to xyz format...")
            for hist_file in tqdm(hist_files):
                if os.path.isfile(hist_file[:-4] + '.xyz'):
                    pass
                else:
                    new_src = hist_file[:-4] + '.xyz'
                    os.rename(hist_file, new_src)

            if not os.path.isdir(PATH + 'shift'):
                os.mkdir(PATH + 'shift')

            print("Getting shifts...")
            for i in tqdm(id_list):
                if os.path.isfile(PATH + 'shifts/' + i + '.txt'):
                    pass
                else:
                    get_shift(path=PATH + 'raw/', name=i)

            mv_Res_without_Shift()

            clean_XYZ()
            clean_XYZ()

            print("Finalizing XYZ file...")
            for i in tqdm(id_list):
                addlineto_xyz(i)

        if analysis:
            xyz = xyz_loader(limit=limit)
            batch_loader = DataLoader(xyz, batch_size=256, shuffle=True)
            start = time.time()
            for idx, batch in enumerate(tqdm(batch_loader)):
                print(batch[0].shape)
            print("{} samples took {} minutes.".format(limit, time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify run (respective .ini file holds all tunable hyperparameters.')
    parser.add_argument('--radius', type=int, default=6, help='Please specify radius.')
    parser.add_argument('--limit', type=int, default=10000, help='Please specify radius.')
    # parser.add_argument('--run', type=int, help='Please specify run ID.')
    # parser.add_argument('--verbose', type=int, default=0, help='Please specify verbose.')
    args = parser.parse_args()
    p = Protos(radius=args.radius, limit=args.limit)
