from Processing.download_pdb import *
from Processing.utils import *
from Processing.loader import *


from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import argparse
import os

PATH = 'data/'


class Protos:
    def __init__(self,
                 radius,
                 download = True,
                 preprocess = True,
                 analysis = False,
                 mode = 'coulomb'):
        self.radius = radius

        if download:
            download_proteins()
            pass

        limit = np.inf
        if preprocess:
            filenames = glob.glob(pathname=PATH+'*.pdb')
            print("Starting to preprocess {} proteins...".format(len(filenames)))
            if limit < len(filenames):
                filenames = filenames[:limit]
            id_list = list(name[5:9] for name in filenames)

            if not os.path.isdir(PATH+'hist'):
                os.mkdir(PATH+'hist')

            print("Creating histograms...")
            for i in tqdm(id_list):
                if self.__check_hist__(i):
                    pass
                else:
                    xyz, types, res = load_pdb(name=i)
                    xyz = np.asarray(xyz)
                    res = np.asarray(res)
                    make_dist_file(i, xyz, types, res, radius=radius)

            print("Cleaning txt files...")
            clean_XYZ()
            clean_XYZ()


            hist_files = glob.glob(PATH+'hist/*.txt')
            print("Casting to xyz format...")
            for hist_file in tqdm(hist_files):
                if os.path.isfile(hist_file[:-4]+'.xyz'):
                    pass
                else:
                    make_xyz_from_txt(hist_file)

            if not os.path.isdir(PATH+'shift'):
                os.mkdir(PATH+'shift')

            """
            if not os.path.isdir(PATH+'hist_noshift'):
                os.mkdir(PATH+'hist_noshift')
            """

            print("Getting shifts...")
            for i in tqdm(id_list):
                if os.path.isfile(PATH+'shifts/'+i+'.txt'):
                    pass
                else:
                    get_shift(i)

            mv_Res_without_Shift()

            print("Finalizing XYZ file...")
            for i in tqdm(id_list):
                addlineto_xyz(i)

        if analysis:
            xyz = xyz_loader(limit = limit)

        batch_loader = DataLoader(xyz, batch_size=256, shuffle=True)
        start = time.time()
        for idx, batch in enumerate(tqdm(batch_loader)):
            print(batch[0].shape)
        print("{} samples took {} minutes.".format(limit, time.time()-start))


    def __check_hist__(self, name):
        hist_files = glob.glob(PATH+'hist/*')
        for h in hist_files:
            if name in h:
                return True
        return False

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify run (respective .ini file holds all tunable hyperparameters.')
    parser.add_argument('--radius', type=int, default=6, help='Please specify radius.')
    # parser.add_argument('--run', type=int, help='Please specify run ID.')
    # parser.add_argument('--verbose', type=int, default=0, help='Please specify verbose.')
    args = parser.parse_args()
    p = Protos(radius=args.radius)