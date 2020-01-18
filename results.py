import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

TITLES = {13003: 'ACN, 10k, batchsize 72',
          14001: 'ACN, 100k, batchsize 72',
          14003: 'ACN, 100k, batchsize 128',
          14004: 'ACN, 10k, batchsize 72',
          14006: 'ACN, 10k, batchsize 128',
          # 14007: 'ACN, 10k, batchsize 128',
          # 14008: 'ACN, 10k, batchsize 128',
          15001: 'ACN, 10k, radius 3, batchsize 72',
          15002: 'ACN, 10k, radius 4, batchsize 72',
          15003: 'ACN, 10k, radius 5, batchsize 72',
          15004: 'ACN, 100, radius 3, batchsize 8',
          15005: 'ACN, 1000, radius 3, batchsize 72',
          15007: 'ACN, 100, radius 4, batchsize 8',
          15008: 'ACN, 100, radius 5, batchsize 8',
          15009: 'ACN, 1000, radius 4, batchsize 8',
          15010: 'ACN, 1000, radius 5, batchsize 8',
          }

# RUNS = [14001, 14003, 14004, 14006, 15001, 15002, 15003, 15004, 15006, 15007, 15008, 15009, 15010]
# RUNS = [14001, 14003, 14004, 14006, 15004, 15007, 15008, 15009, 15010]
RUNS = [13003]


def evaluation(run, number_outliers = 6, print_outliers=True, plot_outliers=False, print_model_specs=True):
    # careful --> have to run test-run with batchsize 1 otw std will be off
    print("")
    print("")
    print("")
    print("")
    print("RUN: ", run)
    print("")
    n, mae, rmse, p, t = load_eval(run)
    avg_mae = np.mean(mae)
    std_mae = np.std(mae)
    avg_rmse = np.mean(rmse)
    std_rmse = np.std(rmse)
    idx_outliers = outliers(p, t, n_outliers=number_outliers)
    if print_outliers:
        print("Outliers: ", [n[idx] for idx in idx_outliers])
    if print_model_specs:
        print(TITLES[run])
    if plot_outliers:
        plot_pred_tar(run, p * 630 * 400, t * 630 * 400, idx_outliers, names=n)
    else:
        plot_pred_tar(run, p * 630 * 400, t * 630 * 400)
    return avg_mae, std_mae, avg_rmse, std_rmse


def outliers(pred, targets, n_outliers, mode='mae'):
    if mode is 'mae':
        diff = cal_mae(pred, targets).flatten()
    if mode is 'rmse':
        diff = cal_rmse(pred, targets).flatten()
    return diff.argsort()[-n_outliers:][::-1]


def plot_pred_tar(run_id, pred, target, idx_outliers=None, names=None, show=False):
    line = np.linspace(0, 630, 1000)
    if idx_outliers is None:
        plt.scatter(pred, target, c='b')
        plt.xlabel(r'real $U_{rt}$ [kcal/mol]', fontsize=14)
        plt.ylabel(r'predicted $U_{rt}$ [kcal/mol]', fontsize=14)
        plt.plot(line, line, 'k-')
    else:
        plt.scatter(pred, target, c='b', )
        plt.scatter(pred[idx_outliers], target[idx_outliers], c='r')
        plt.xlabel(r'real $U_{rt}$ [kcal/mol]', fontsize=14)
        plt.ylabel(r'predicted $U_{rt}$ [kcal/mol]', fontsize=14)
        for i, (x, y) in enumerate(zip(pred[idx_outliers], target[idx_outliers])):
            label = names[idx_outliers[i]][-10:-4]
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
        plt.plot(line,line,'k-')
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.title(TITLES[run_id], fontsize=18)
    plt.savefig('plots/eval_'+str(run_id)+'.png', dpi=1200)
    if show:
        plt.show()
    plt.close()
    return


def load_eval(run):
    file = 'models/run_' + str(run // 100) + '/model_' + str(run) + '_eval.txt'
    f = open(file, 'rb')
    print('Reading file', file)
    results = pickle.load(f)
    predictions = results['predictions']
    targets = results['targets']
    names = results['names']
    p = []
    t = []
    n = []
    mae = []
    rmse = []
    for batch in zip(predictions, targets, names):
        for j in range(len(batch[0])):
            p.append(batch[0][j])
            t.append(batch[1][j])
            n.append(batch[2][j])
            mae.append(cal_mae(batch[0][j], batch[1][j]))
            rmse.append(cal_rmse(batch[0][j], batch[1][j]))
    f.close()
    return n, mae, rmse, np.asarray(p), np.asarray(t)


def cal_mae(x, y):
    return np.abs(x - y)


def cal_rmse(x, y):
    return np.sqrt((x - y) ** 2)


if __name__ == '__main__':
    my_run = [14001]
    for run in RUNS:
        eval_ = evaluation(run)
        print(('Run {}: MAE average: {}  ---   MAE standard deviation: {}'
               '\n        RMSE average: {}  ---  RMSE standard deviation: {}')
              .format(run, eval_[0], eval_[1], eval_[2], eval_[3]))
