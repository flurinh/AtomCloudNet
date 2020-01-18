import os


# RUNS = [14001, 14003, 14004, 14006, 15001, 15002, 15003, 15004, 15006, 15007, 15008, 15009, 15010]
RUNS = [15001, 15002, 15003]

if __name__ == '__main__':
    for run in RUNS:
        os.system('bsub -W 1:00 -R "rusage[mem=10000]" python sandbox.py --run ' + str(run) + ' --mode 1')