import pickle
import argparse
import numpy as np


def evaluation(run=13002):
    # careful --> have to run test-run with batchsize 1 otw std will be off
    m, r = load(run)
    avg_mae = np.mean(m)
    std_mae = np.std(m)
    avg_rmse = 0#np.mean(r)
    std_rmse = 0#np.std(r)
    return avg_mae, std_mae, avg_rmse, std_rmse


def load(run):
    file = '../models/run_'+str(run//100)+'/model_'+str(run)+'_eval.txt'
    f = open(file, 'rb')
    print('opening file', f)
    results = pickle.load(f)
    print("pickle loaded", results)
    rmse_losses = results['rmse_losses']
    mae_losses = results['mae_losses']
    f.close()
    return mae_losses, rmse_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify setting (generates all corresponding .ini files).')
    parser.add_argument('--run', type=int, default=13002)
    args = parser.parse_args()
    output = evaluation(args.run)
    print(('Run {}: MAE average: {}  ---   MAE standard deviation: {}'
          '\n        RMSE average: {}  ---  RMSE standard deviation: {}')
          .format(args.run, output[0], output[1], output[2], output[3]))