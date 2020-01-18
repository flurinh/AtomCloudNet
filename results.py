import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


# A selection of configuration we explored. Feel free to write your own config files.
TITLES = {13003: 'ACN, 10k, radius 3, batchsize 72',
          15002: 'ACN, 10k, radius 4, batchsize 72',
          15003: 'ACN, 10k, radius 5, batchsize 72',
          15004: 'ACN, 100, radius 3, batchsize 8',
          15005: 'ACN, 1000, radius 3, batchsize 8',
          15007: 'ACN, 100, radius 4, batchsize 8',
          15008: 'ACN, 100, radius 5, batchsize 8',
          15009: 'ACN, 1000, radius 4, batchsize 8',
          15010: 'ACN, 1000, radius 5, batchsize 8',
          }

RUNS = [13003, 15002, 15003, 15004, 15005, 15007, 15008, 15009, 15010]


def evaluation(run, number_outliers, print_outliers=True, plot_outliers=False, print_model_specs=True):
    """
    Evaluation of run corresponding to configuration 'run'.
    :param run:
    :param number_outliers: Selects number_outliers samples with the highest MAE
    :param print_outliers: whether to print them out
    :param plot_outliers: whether to include them marked and with name in the plot
    :param print_model_specs: whether to print model specs
    :return:
    """
    print("")
    print("")
    print("")
    print("")
    print("RUN: ", run)
    print("")
    n, mae, rmse, p, t = load_eval(run)
    idx_outliers = outliers(p, t, n_outliers=number_outliers)
    if print_outliers:
        print("Outliers: ", [n[idx] for idx in idx_outliers])
    if print_model_specs:
        print(TITLES[run])
    if plot_outliers:
        plot_pred_tar(run, p * 630 * 400, t * 630 * 400, idx_outliers, names=n)
    else:
        plot_pred_tar(run, p * 630 * 400, t * 630 * 400)
    return mae, np.NaN, rmse, np.NaN


def outliers(pred, targets, n_outliers, mode='mae'):
    if mode is 'mae':
        diff = np.abs(pred - targets).flatten()
    return diff.argsort()[-n_outliers:][::-1]


def plot_pred_tar(run_id, pred, target, idx_outliers=None, names=None, show=False):
    line = np.linspace(0, 630 * 400, 1000)
    plt.figure(figsize=(20, 10))
    if idx_outliers is None:
        plt.scatter(pred, target, c='b')
        plt.xlabel(r'real $U_{rt}$ [kcal/mol]', fontsize=24)
        plt.ylabel(r'predicted $U_{rt}$ [kcal/mol]', fontsize=24)
        plt.plot(line, line, 'k-')
    else:
        plt.scatter(pred, target, c='b', )
        plt.scatter(pred[idx_outliers], target[idx_outliers], c='r')
        plt.xlabel(r'real $U_{rt}$ [kcal/mol]', fontsize=24)
        plt.ylabel(r'predicted $U_{rt}$ [kcal/mol]', fontsize=24)
        for i, (x, y) in enumerate(zip(pred[idx_outliers], target[idx_outliers])):
            label = names[idx_outliers[i]][-10:-4]
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
        plt.plot(line,line,'k-')
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.title(TITLES[run_id], fontsize=28)
    plt.savefig('plots/eval_'+str(run_id)+'.png', dpi=1200)
    if show:
        plt.show()
    plt.close()
    return


def load_eval(run):
    """
    Loads the evaluation file created by running "sandbox.py --run *run* --mode 1" and calculates errors
    :param run: *run*-id
    :return:
    n: names of each molecule
    mae_: mean absolute error
    rmse_: root mean squared error
    p: predictions
    t: targets
    """
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
    for batch in zip(predictions, targets, names):
        for j in range(len(batch[0])):
            p.append(batch[0][j])
            t.append(batch[1][j])
            n.append(batch[2][j])
    p_ = np.asarray(p)
    t_ = np.asarray(t)
    mae_ = mean_absolute_error(t_, p_)
    rmse_ = np.sqrt(mean_squared_error(t_, p_))
    f.close()
    return n, mae_, rmse_, p_, t_


def cal_mae(x, y):
    return mean_absolute_error(y, x)


def cal_rmse(x, y):
    return mean_squared_error(y, x)


if __name__ == '__main__':
    """
    After specifying a list of configuration ids corresponding to trained and evaluated models we can plot and review
    the results by running this file.
    """
    for run in RUNS:
        eval_ = evaluation(run, number_outliers=6)
        print(('Run {}: MAE average:  {} '
               '\n           RMSE average: {} ')
              .format(run, eval_[0], eval_[2]))
