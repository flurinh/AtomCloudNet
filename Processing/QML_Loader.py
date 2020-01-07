import random
import numpy as np
from copy import deepcopy
import qml
from math import sqrt
from sklearn.metrics import mean_squared_error
from qml.math import cho_solve
from qml.kernels import laplacian_kernel
from qml.representations import *
from qml.fchl import get_local_kernels
from tqdm import tqdm
import sys
import glob

np.set_printoptions(threshold=sys.maxsize)

path = "../data/"
PATH = path

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

def get_FCHL(PATH):

    print("\n -> calculating dict with properties")

    data = get_energies(PATH + "trainUrt.txt")
    data2 = get_energies(PATH + "testUrt.txt")

    mols = []
    mols_test = []

    for xyz_file in sorted(data.keys()):
        mol = qml.Compound()
        mol.read_xyz(PATH + "QM9Train/" + xyz_file)
        mol.properties = data[xyz_file]
        mols.append(mol)
    for xyz_file in sorted(data2.keys()):
        mol = qml.Compound()
        mol.read_xyz(PATH + "QM9Test/" + xyz_file)
        mol.properties = data2[xyz_file]
        mols_test.append(mol)

    print("\n -> generate FCHL representation")
    for mol in tqdm(mols):
        mol.generate_fchl_representation()
    for mol in tqdm(mols_test):
        mol.generate_fchl_representation()

    X = np.asarray([mol.representation for mol in mols])
    X_test = np.asarray([mol.representation for mol in mols_test])

    Yprime  = np.asarray([mol.properties for mol in mols])
    Ytest  = np.asarray([mol.properties for mol in mols_test])

    return X, X_test, Yprime, Ytest




def get_CM(PATH):

    print("\n -> calculating dict with properties")

    data = get_energies(PATH + "trainUrt.txt")
    data2 = get_energies(PATH + "testUrt.txt")

    mols = []
    mols_test = []

    for xyz_file in sorted(data.keys()):
        mol = qml.Compound()
        mol.read_xyz(PATH + "QM9Train/" + xyz_file)
        mol.properties = data[xyz_file]
        mols.append(mol)
    for xyz_file in sorted(data2.keys()):
        mol = qml.Compound()
        mol.read_xyz(PATH + "QM9Test/" + xyz_file)
        mol.properties = data2[xyz_file]
        mols_test.append(mol)

    print("\n -> generate CM representation")
    for mol in tqdm(mols):
        mol.generate_coulomb_matrix(size=23)
    for mol in tqdm(mols_test):
        mol.generate_coulomb_matrix(size=23)

    X = np.asarray([mol.representation for mol in mols])
    X_test = np.asarray([mol.representation for mol in mols_test])

    Yprime  = np.asarray([mol.properties for mol in mols])
    Ytest  = np.asarray([mol.properties for mol in mols_test])

    return X, X_test, Yprime, Ytest



def get_en_file(PATH):
    filenames = glob.glob(PATH + "QM9Test/*.xyz")

    for f in sorted(filenames):
        fi = open(f)
        lines = fi.readline()
        l2 = fi.readline()
        fi.close()
        tokens = l2.split()
        print(f[16:], tokens[14])


#get_en_file(PATH)
#get_FCHL(PATH)
get_CM(PATH)