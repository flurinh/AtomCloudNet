import random
import numpy as np
from copy import deepcopy
import qml
from math import sqrt
from sklearn.metrics import mean_squared_error
from qml.math import cho_solve
from qml.representations import *
from tqdm import tqdm
from qml.kernels import laplacian_kernel
from qml.kernels import gaussian_kernel
from qml.kernels import linear_kernel
import sys

np.set_printoptions(threshold=sys.maxsize)

"""
Function to get binding energies.
"""

def get_energies(filename):

    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    energies = dict()

    for line in lines:
        tokens = line.split()
        xyz_name = tokens[0]
        Ebind = float(tokens[1])
        energies[xyz_name] = Ebind

    return energies

"""
Generating dict with binding energies for xyz files.
"""

if __name__ == "__main__":
    print("\n -> make dict of binding energy per molecules")

    data = get_energies("../data/trainDipole.txt")
    data2 = get_energies("../data/testDipole.txt")
    mols = []
    mols_test = []
    for xyz_file in tqdm(sorted(data.keys())):
        mol = qml.Compound()
        mol.read_xyz("../data/QM9Train/" + xyz_file)
        mol.properties = data[xyz_file]
        mols.append(mol)
    for xyz_file in tqdm(sorted(data2.keys())):
        mol = qml.Compound()
        mol.read_xyz("../data/QM9Test/" + xyz_file)
        mol.properties = data2[xyz_file]
        mols_test.append(mol)

    print("\n -> generate Coulomb Matrix representation")
    for mol in tqdm(mols):
        mol.generate_coulomb_matrix(size=29)
    for mol in tqdm(mols_test):
        mol.generate_coulomb_matrix(size=29)
    
    """
    Setting hyperparameters.
    """
    
    N = [100, 1000, 2000, 5000, 10000]
    #N = [10000]
    nModels = 10
    total = 10000

    random.seed(667)

    sigmas   =  [0.2*2**i for i in range(8, 12)]
    sigmas = [104857.6]

    X = np.asarray([mol.representation for mol in mols])
    X_test = np.asarray([mol.representation for mol in mols_test])

    Yprime  = np.asarray([mol.properties for mol in mols])
    Ytest  = np.asarray([mol.properties for mol in mols_test])

    print("\n -> calculating laplacian kernels")
    
    """
    Calculating kernels, cross validation and predictions.
    """
    
    for j in range(len(sigmas)):
        K = laplacian_kernel(X, X, sigmas[j])
        K_test = laplacian_kernel(X, X_test, sigmas[j])
        for train in N:
            maes    = []
            test = total - train
            for i in range(nModels):
                split = list(range(total))
                random.shuffle(split)

                training_index = split[:train]
                test_index = split[-test:]

                #print(len(training_index))
                #print(len(test_index))

                Y = Yprime[training_index]
                Ys = Yprime[test_index]

                C = deepcopy(K[training_index][:, training_index])
                C[np.diag_indices_from(C)] += 10.0 ** (-7.0)

                alpha = cho_solve(C, Y)

                Yss = np.dot((K_test[training_index]).T, alpha)
                diff = Yss - Ytest
                mae = np.mean(np.abs(diff))
                maes.append(mae)
                rms = sqrt(mean_squared_error(Yss, Ytest))
            s = np.std(maes) / np.sqrt(nModels)

            print(str(sigmas[j]) + "\t" + str(train) + "\t" + str(sum(maes) / len(maes)) + "\t" + str(s) + "\t" + str(rms))

