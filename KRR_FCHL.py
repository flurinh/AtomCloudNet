import random
import numpy as np
from copy import deepcopy
import qml
from math import sqrt
from sklearn.metrics import mean_squared_error
from qml.math import cho_solve
from qml.representations import *
from qml.kernels import laplacian_kernel, gaussian_kernel
from qml.fchl import get_local_kernels
from tqdm import tqdm

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


if __name__ == "__main__":
    print("\n -> load binding energies")

    data = get_energies("data/trainUrt.txt")
    data2 = get_energies("data/testUrt.txt")
    mols = []
    mols_test = []
    for xyz_file in tqdm(sorted(data.keys())):
        mol = qml.Compound()
        mol.read_xyz("data/QM9Train/" + xyz_file)
        mol.properties = data[xyz_file]
        mols.append(mol)
    for xyz_file in tqdm(sorted(data2.keys())):
        mol = qml.Compound()
        mol.read_xyz("data/QM9Test/" + xyz_file)
        mol.properties = data2[xyz_file]
        mols_test.append(mol)

    mbtypes = get_slatm_mbtypes([mol.nuclear_charges for mol in mols + mols_test])

    print("\n -> generate representation")
    for mol in tqdm(mols):
        mol.generate_slatm(mbtypes, local=False)
    for mol in tqdm(mols_test):
        mol.generate_slatm(mbtypes, local=False)

    N = [100, 1000, 2000, 5000, 9986]
    nModels = 10
    total = 9986

    sigmas   =  [0.2*2**i for i in range(8, 22)]

    X = np.asarray([mol.representation for mol in mols])
    X_test = np.asarray([mol.representation for mol in mols_test])

    Yprime  = np.asarray([mol.properties for mol in mols])
    Ytest  = np.asarray([mol.properties for mol in mols_test])

    """np.save("data/krr/trainingFCHL", X)
    np.save("data/krr/testFCHL", X_test)"""

    print("\n -> calculating kernels")

    random.seed(667)

    print("\n -> calculating cross validation and predictions")
    for j in tqdm(range(len(sigmas))):

        K = gaussian_kernel(X, X, sigmas[j])
        K_test = gaussian_kernel(X, X_test, sigmas[j])

        for train in N:
            test = total - train
            maes = []
            for i in range(nModels):
                split = list(range(total))
                random.shuffle(split)

                training_index = split[:train]
                test_index = split[-test:]

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

