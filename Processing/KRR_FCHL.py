import random
import numpy as np
from copy import deepcopy
import qml
from math import sqrt
from sklearn.metrics import mean_squared_error
from qml.math import cho_solve
from qml.representations import *
from qml.fchl import get_local_kernels
from tqdm import tqdm
import sys
from QML_Loader import *

np.set_printoptions(threshold=sys.maxsize)

path = "../data/"
PATH = path

X, X_test, Yprime, Ytest = get_FCHL(PATH)

N = [100, 1000, 2000, 5000]
nModels = 10
total = 10000

sigma = [0.2 * 2 ** i for i in range(20)]

random.seed(667)

print("\n -> calculating FCHL Kernels")

K = get_local_kernels(X, X, sigma, cut_distance=10.0)
K_test = get_local_kernels(X, X_test, sigma, cut_distance=10.0)

print("\n -> calculating FCHL predictions")

for j in tqdm(range(len(sigma))):
    for train in tqdm(N):
        maes    = []
        test = total - train
        for i in tqdm(range(nModels)):
            split = list(range(total))
            random.shuffle(split)

            training_index  = split[:train]
            test_index      = split[-test:]

            Y = Yprime[training_index]
            Ys = Yprime[test_index]

            C = deepcopy(K[j][training_index][:, training_index])
            C[np.diag_indices_from(C)] += 10.0**(-7.0)
            alpha = cho_solve(C, Y)

            Yss = np.dot((K_test[j][training_index]).T, alpha)

            diff = Yss  - Ytest
            mae = np.mean(np.abs(diff))
            maes.append(mae)
            rms = sqrt(mean_squared_error(Yss, Ytest))
            rms = rms * 3.7
        s = np.std(maes) / np.sqrt(nModels)

        print(str(sigma[j]) + "\t" + str(train) + "\t" + str(sum(maes) / len(maes)) + "\t" + str(s) + "\t" + str(rms))
