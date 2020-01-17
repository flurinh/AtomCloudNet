import pickle
import argparse
import numpy as np


def evaluation(run=13002):
    # careful --> have to run test-run with batchsize 1 otw std will be off
    m, r = load_eval(run)
    avg_mae = np.mean(m)
    std_mae = np.std(m)
    avg_rmse = 0  # np.mean(r)
    std_rmse = 0  # np.std(r)
    return avg_mae, std_mae, avg_rmse, std_rmse


def load_eval(run):
    file = 'models/run_' + str(run // 100) + '/model_' + str(run) + '_eval.txt'
    f = open(file, 'rb')
    print('opening file', f)
    results = pickle.load(f)
    print("pickle loaded", results)
    rmse_losses = results['rmse_losses']
    mae_losses = results['mae_losses']
    f.close()
    return mae_losses, rmse_losses


if __name__ == '__main__':
    run = 14001
    output = {}
    for i in range(9):
        eval_ = evaluation(run + i)
        output.update({str(run + i): eval_})
        print(('Run {}: MAE average: {}  ---   MAE standard deviation: {}'
               '\n        RMSE average: {}  ---  RMSE standard deviation: {}')
              .format(run+i, eval_[0], eval_[1], eval_[2], eval_[3]))
