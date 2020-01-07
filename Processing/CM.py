import random
import numpy as np
from copy import deepcopy
import qml
from qml.math import cho_solve
from qml.representations import *
from qml.kernels import laplacian_kernel


def get_energies(filename):
    """ returns dict with energies for xyz files
    """
    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    energies = dict()

    for line in lines:
        tokens = line.split()
        xyz_name = tokens[0]
        Ebind = float(tokens[1]) #*627.509
        energies[xyz_name] = Ebind

    return energies

if __name__ == "__main__":

    # get binding energies
    """data = get_energies("../00Learn/Train/binding.txt")#("train_total.txt")
    data2 = get_energies("../00Learn/Test/binding.txt")#("test_total.txt")"""

    mols = []
    mols_test = []

    # read molecules    first trainingset then testset
    for xyz_file in sorted(data.keys()):
        mol = qml.Compound()
        mol.read_xyz("../QM9/dsgdb9nsd_000009" + xyz_file)
        mol.properties = data[xyz_file]
        mols.append(mol)
        print(mols)
        
    """for xyz_file in sorted(data2.keys()):
        mol = qml.Compound()
        mol.read_xyz("../00Learn/Test/" + xyz_file)
        mol.properties = data2[xyz_file]
        mols_test.append(mol)"""



    print("\n -> calculating the Representation ")
    for mol in mols:
        mol.generate_coulomb_matrix(size=20)
    """for mol in mols_test:
        mol.generate_coulomb_matrix(size=90)"""

    """N       = [8, 16, 32, 64, 128] #[16,32, 64,128] #[25,50,100,200,400,800,1621] #[1621] #
    nModels = 10
    total   = 128

    sigma   =  [64000] #[102.4] #[0.2*2**i for i in range(16)] #[819.2]#[1600]
    X = np.array([mol.representation for mol in mols])
    X_test = np.array([mol.representation for mol in mols_test])


  # properties
    Yprime  = np.asarray([ mol.properties for mol in mols ])
    Ytest  = np.asarray([ mol.properties for mol in mols_test ])

    K       = laplacian_kernel(X, X, sigma)
    K_test  = laplacian_kernel(X, X_test, sigma)

    random.seed(667)

  # outer loop only used Kernels that contain more than one sigma
    for j in range(len(sigma)):
    # loop over training set sizes
        for train in N:
            maes    = []
            test = total - train
      # loop over # of machines to train (cross valiidation)
            for i in range(nModels):
                split = list(range(total))
                random.shuffle(split)

                training_index  = split[:train]
                test_index      = split[-test:]

                Y = Yprime[training_index]
                Ys = Yprime[test_index]

                C = deepcopy(K[training_index][:,training_index])
                C[np.diag_indices_from(C)] += 10.0**(-7.0)

                alpha = cho_solve(C, Y)

                Yss = np.dot((K_test[training_index]).T, alpha)
                diff = Yss  - Ytest
                mae = np.mean(np.abs(diff))
                maes.append(mae)
            s = np.std(maes)/np.sqrt(nModels)

      # print the results to the terminal (thez will not be stored!)

            print(str(sigma[j]) + "\t" + str(train) + "\t" + str(sum(maes)/len(maes)) + " " + str(s))

            #print(Yss)
            x = np.asarray(Yss)
            y = np.asarray(Ys)
            print(Ytest)
            print(x)
            #print(y)
"""
