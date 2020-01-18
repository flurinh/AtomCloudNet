import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

TITLES = {}

def evaluation(run):
    # careful --> have to run test-run with batchsize 1 otw std will be off
    n, mae, rmse, p, t = load_eval(run)
    avg_mae = np.mean(mae)
    std_mae = np.std(mae)
    avg_rmse = np.mean(rmse)
    std_rmse = np.std(rmse)
    idx_outliers = outliers(n, p, t, n_outliers = 10)
    plot_pred_tar(p, t, idx_outliers, n)
    return avg_mae, std_mae, avg_rmse, std_rmse


def outliers(pred, targets, n_outliers, mode='mae'):
    if mode is 'mae':
        diff = cal_mae(pred, targets)
    if mode is 'rmse':
        diff = cal_rmse(pred, targets)
    return (-diff).argsort()[:n_outliers]


def plot_pred_tar(title, runpred, target, idx_outliers=None, names=None):
    w = [0.66715174, 1.02072885, 0.97581753, 0.99144229, 1.06441079, 1.02803944, 1.05230331, 0.87701009, 0.99043134,
         1.07949002, 0.79217865, 0.84969539, 0.98805825, 0.95700719, 0.95026552, 0.91982687, 0.80053463, -0.13962154,
         0.00486584, 0.15993667, 0.04408516, 0.00264467, -0.00164806, 0.15121694, 0.08060598, 0.09944911, -0.06250272,
         0.01597039, -0.02812696, 0.03139001, -0.10828583, 0.35942502, 0.03599929, 0.03066193, -0.12643404, 0.04610165,
         -0.0046145]
    v = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    w = [(i * 3.52 + 4.48) for i in w]
    v = [(i * 3.52 + 4.48) for i in v]
    plt.scatter(pred, target, c="b", label ="")
    plt.xlabel("predicted", fontsize = 14)
    plt.ylabel("real", fontsize=14)
    plt.tick_params(axis='', labelsize = 12)
    plt.tick_params(axis=‘y’, labelsize = 12)
    plt.title(title, fontsize = 18)
    plt.legend(loc=‘right’)
    plt.show()
    pass



def load_eval(run):
    file = 'models/run_' + str(run // 100) + '/model_' + str(run) + '_eval.txt'
    f = open(file, 'rb')
    print('Reading file', file)
    results = pickle.load(f)
    predictions = results['predictions']
    targets = results['argets']
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
    run = 14007
    output = {}
    eval_ = evaluation(run)
    output.update({str(run): eval_})
    print(('Run {}: MAE average: {}  ---   MAE standard deviation: {}'
           '\n        RMSE average: {}  ---  RMSE standard deviation: {}')
          .format(run, eval_[0], eval_[1], eval_[2], eval_[3]))
