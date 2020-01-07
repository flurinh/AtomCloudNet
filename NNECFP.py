import random
import numpy as np
import sys
from sklearn.metrics import mean_squared_error
from math import sqrt
from copy import deepcopy
from qml.math import cho_solve
from qml.representations import *
from qml.kernels import laplacian_kernel
from tqdm import tqdm, trange
import glob
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import torch
import torch.nn as nn

from configparser import ConfigParser
import argparse
import ast
import os

np.set_printoptions(threshold=sys.maxsize)

path = "../data/"
PATH = path

# https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets


# self.['SETTING']['layers'] = ast.literal_eval(config.get("section", "option"))


def get_config(id = 1):
    path = "config/config_"
    config_file = path + str(id).zfill(5) + ".ini"
    print("Loading config files.. " + config_file)
    parser = ConfigParser()
    parser.read(config_file)
    model = parser['SETTING']['model']
    lr = float(parser['SETTING']['lr'])
    momentum = float(parser['SETTING']['momentum'])
    dropout = float(parser['SETTING']['dropout'])
    patience = int(parser['SETTING']['patience'])
    pfactor = float(parser['SETTING']['pfactor'])
    epochs = int(parser['SETTING']['epochs'])
    batchsize = int(parser['SETTING']['batchsize'])
    architectures = list(ast.literal_eval(parser.get("SETTING", "layers")))
    return model, lr, momentum, dropout, patience, pfactor, epochs, batchsize, architectures


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
        Ebind = float(tokens[1])
        energies[xyz_name] = Ebind
    return energies


class CoulombNet(nn.Module):
    """
    Simple feedforward architecture neural network. Baseline module.
    """

    def __init__(self, in_shape, layers, dropout):
        super(CoulombNet, self).__init__()
        self.last_layer_out = None
        self.coul_blocks = nn.ModuleList()
        in_channel = in_shape
        for _ in range(len(layers)):
            self.coul_blocks.append(nn.Linear(in_channel, layers[_]))
            self.coul_blocks.append(nn.BatchNorm1d(layers[_])) # TODO: if batch larger than one, always normalized
            self.coul_blocks.append(nn.Dropout(dropout))    # TODO: dropout
            self.coul_blocks.append(nn.Sigmoid())       # TODO: sigmoid if labels btw 0-1   0-lim(infinite) Relu, -1-1 Tanh
            in_channel = layers[_]
        self.final = nn.Linear(in_channel, 1)
        self.fact = torch.nn.Sigmoid()  # self.fact = nn.Sigmoid() # TODO: final layer sigmoid

    def forward(self, x):
        for _, block in enumerate(self.coul_blocks):
            x = block(x)
        self.last_layer_out = x.clone().detach().numpy()
        return self.fact(self.final(x))


if __name__ == "__main__":
    print("\n -> generate dict with binding data")

    hyperparam = get_config(1)

    # TODO: Bioactivity in Array
    data = get_energies("../data/TrainSmiles/binding.txt")
    data2 = get_energies("../data/TestSmiles/binding.txt")

    en = []
    en2 = []
    for i in tqdm(sorted(data.keys())):
        properties = data[i]
        en.append(properties)
    for i in tqdm(sorted(data2.keys())):
        properties = data2[i]
        en2.append(properties)

    Yprime = np.asarray(en)
    Ytest = np.asarray(en2)

    print("\n -> generate Representation")

    # TODO: Representation in Array

    trainfiles = glob.glob(PATH + "TrainSmiles/" + "*.smi")
    mols = []
    sorted_trainfiles = sorted(trainfiles)
    for trainfile in tqdm(sorted_trainfiles):
        f = open(trainfile)
        tokens = f.readline().split()
        f.close()
        smiles = []
        smiles.append(tokens[0])
        for smile in smiles:
            num_bit = 1024
            radius = 2
            ecfp = rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), radius,
                                                                  nBits=num_bit)  # .ToBitString()
            # maccskeys = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smile))      #.ToBitString()
            mols.append(np.asarray(ecfp))

    testfiles = glob.glob(PATH + "TestSmiles/" + "*.smi")
    mols_test = []
    sorted_testfiles = sorted(testfiles)
    for testfile in tqdm(sorted_testfiles):
        f = open(testfile)
        tokens = f.readline().split()
        f.close()
        smiles = []
        smiles.append(tokens[0])
        for smile in smiles:
            num_bit = 1024
            radius = 2
            ecfp = rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), radius,
                                                                  nBits=num_bit)  # .ToBitString() if wished as sting
            # maccskeys = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smile))  #.ToBitString()
            e = np.asarray(ecfp)
            mols_test.append(np.asarray(e))

    X = np.asarray(mols)
    X_test = np.asarray(mols_test)

    """print(X_test.shape)
    print(Ytest.shape)
    print(X.shape)
    print(Yprime.shape)"""

    """sigmas = [40]
    for j in range(len(sigmas)):
        K = laplacian_kernel(X, X, sigmas[j])
        K_test = laplacian_kernel(X_test, X, sigmas[j], )

    print(K.shape)
    print(K_test.shape)"""

    # TODO: Load data into nn and run nn

    model = CoulombNet(1024, layers=hyperparam[8], dropout=hyperparam[3])

    print("Model :", model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number trainable parameters:", params)

    criterion = nn.MSELoss()  # MSELoss() L1Loss()

    mae_crit = nn.L1Loss()
    opt = torch.optim.SGD(model.parameters(), lr=hyperparam[1], momentum=hyperparam[2])  # torch.optim.SGD ADAM

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=hyperparam[5], patience=hyperparam[4], verbose=True) # TODO: adjust patience and factor , 0.5 and 50

    epochs = hyperparam[6]
    batch_size = hyperparam[7]
    niter = X.shape[0] // batch_size
    print("Number iterations:", niter)
    prediction = None
    for e in trange(epochs):
        ids = np.random.shuffle(np.arange(X.shape[0]))
        X_ = X[ids].squeeze()
        Y_ = Yprime[ids].squeeze()

        mae_epoch_loss = 0
        mean_epoch_loss = 0
        epoch_loss = 0
        model.train()
        for i in trange(niter):
            x_in = torch.tensor(X_[i * batch_size:(i + 1) * batch_size, :], dtype=torch.float)
            y_in = torch.tensor(Y_[i * batch_size:(i + 1) * batch_size], dtype=torch.float)
            opt.zero_grad()
            prediction = model(x_in)
            loss = criterion(prediction.squeeze(), y_in)
            mae_loss = mae_crit(prediction.squeeze(), y_in).item()
            loss.backward()
            loss_ = loss.item()
            epoch_loss += loss_
            mae_epoch_loss += mae_loss
            opt.step()
        mae_epoch_loss = mae_epoch_loss / (niter)
        mean_epoch_loss = epoch_loss / (niter)
        scheduler.step(mean_epoch_loss, e)
        print("Training MAE:", mae_epoch_loss)
        print("Training Loss (MSE):", mean_epoch_loss)
        mean_epoch_loss = 0
        epoch_loss = 0

        if e % 20 == 0:
            model.eval()
            xt_in = torch.tensor(X_test, dtype=torch.float)
            yt_in = torch.tensor(Ytest, dtype=torch.float)
            prediction = model(xt_in)
            loss = criterion(prediction.squeeze(), yt_in)
            mae_loss = mae_crit(prediction.squeeze(), yt_in).item()
            mean_epoch_loss = loss.item()
            p = prediction.detach().cpu().numpy().squeeze()
            p2 = yt_in.numpy()
            # print(np.asarray(model.last_layer_out))
            diff = p - p2
            mae = np.mean(np.abs(diff))
            rmse = np.sqrt(mean_epoch_loss)
            print()
            print(" ========== TEST ===========")
            print("Kenneth mae:", mae)
            print("torch MAE:", mae_loss)
            print("Kenneth rmse:", rmse)
            print(" ===========================")
            """print(Ytest)
            print(prediction)"""
            torch.save(model, 'model_ecfp.pt')
