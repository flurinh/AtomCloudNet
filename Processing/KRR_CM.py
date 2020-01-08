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
    for j in range(len(sigmas)):
        K = laplacian_kernel(X, X, sigmas[j])
        K_test = laplacian_kernel(X, X_test, sigmas[j])
        """K = gaussian_kernel(X, X, sigmas[j])
        K_test = gaussian_kernel(X, X_test, sigmas[j])
        K = linear_kernel(X, X)
        K_test = linear_kernel(X, X_test)"""
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


"""
Laplacian Kernel: in Hartree,     Ha * 630 = kcal/mol                                           kcal/mol,               eV
104857.6        100     6.146355701246402       0.15217299649321853     9.991812711774989       3856.8972569057287      167.25093738254066
104857.6        1000    0.8118483549306534      0.01623470576872112     1.6665485184785847      509.44264298281587      22.091529513511627
104857.6        2000    0.40411114798679526     0.005203141096532805    0.9539171216837397      253.58362807398674      10.996429688216733
104857.6        5000    0.16293243399843274     0.0014784175028615585   0.7166997317045918      102.24166779382695      4.433619471572519
104857.6        10000   0.08099263293528651     1.0726930952137766e-12  0.6572137753487519      50.82365534658731       2.203922850862622

Gap:
1638.4  100     0.03268800878699056     0.000299921394724423    0.039927098608438315
1638.4  1000    0.023057407507521728    0.0001349568149688998   0.029963938168411516
1638.4  2000    0.020318439599906528    8.162990020374851e-05   0.02613664166719381
1638.4  5000    0.017238478311976967    2.8321184043250226e-05  0.022571437743146905
1638.4  10000   0.015066072390620428    3.0206746412445463e-16  0.019964198719956246


Dipole:
819.2   100     1.0525329901255742      0.007936358011057782    1.4571495812183775
819.2   1000    0.8936225927368733      0.003861948227591202    1.2310726271403039
819.2   2000    0.8368179786879102      0.002535907060705981    1.163759436155043
819.2   5000    0.7533921168275508      0.001442195612427025    1.054219429059037
819.2   10000   0.6870190733703806      3.5146581937170504e-15  0.9647974775209867

"""
