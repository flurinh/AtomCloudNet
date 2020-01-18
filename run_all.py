from configparser import ConfigParser
import argparse
import time
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify setting (generates all corresponding .ini files).')
    parser.add_argument('--setting', type=int, default=1)  # setting == 0 if you want to run coulombnet
    args = parser.parse_args()
    if args.setting == 1:
        path = 'config_ACN/'
    else:
        path = 'config/'
    files = [name for name in os.listdir(path)]
    num_runs = len(files) - 1
    print("Running a total of {} jobs!".format(num_runs))

    if args.setting == 0:
        for i in range(24):
            ini_id = i + 1
            print("Running configuration N°" + str(ini_id).zfill(5))
            os.system('bsub -W 24:00 -R "rusage[mem=16382]" python nnCM.py --run ' + str(ini_id))

    if args.setting == 1:
        for i, f in enumerate(files):
            ini_id = 15001 + i
            if ini_id > 15000:
                print("Running configuration N°" + str(ini_id).zfill(5))
                os.system('bsub -W 12:00 -R "rusage[mem=10000, ngpus_excl_p=1]" python sandbox.py --run ' + str(ini_id) + '')
                time.sleep(1)
                # ssh -NfL localhost:17779:localhost:7779 hidberf@login.leonhard.ethz.ch
                # module load eth_proxy gcc/6.3.0 python_gpu/3.7.4
                # scp -r hidberf@login.leonhard.ethz.ch:/cluster/home/hidberf/Protos/runs/run_140 /Users/modlab/PycharmProjects/Protos/runs
                # scp -r "C:/Users/Flurin Hidber/PycharmProjects/Protos/data/pkl/data_100000.pkl" hidberf@login.leonhard.ethz.ch:/cluster/home/hidberf/Protos/data/pkl
                # bsub -n 1 -Is -W 1:00 -R "rusage[mem=20000, ngpus_excl_p=2]" python sandbox.py --run 14001
                # os.system('bsub -W 24:00 python sandbox.py --run ' + str(ini_id))
                # os.system('python sandbox.py --run ' + str(int(ini_id)))
                # run: module load python_gpu/3.7.1 gcc/6.3.0
                # conda install -c psi4 gcc-5
                # https://scicomp.ethz.ch/wiki/Using_the_batch_system
                # conda env export | grep -v "^prefix: " > environment.yml
                # conda env create -f environment.yml