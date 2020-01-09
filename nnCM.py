import numpy as np
import sys, random
import qml
from qml.representations import *
from qml.kernels import laplacian_kernel
from tqdm import tqdm, trange
from Architectures.CoulombNet import CoulombNet
from utils import *
import torch
import torch.nn as nn

np.set_printoptions(threshold=sys.maxsize)
path = "data/"
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Specify setting (generates all corresponding .ini files).')
    parser.add_argument('--run', type=int, default=1)
    args = parser.parse_args()

    print("\n -> make dict of binding energy of molecules")

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

    print("\n -> generate Coulomb Matrix representation")
    for mol in tqdm(mols):
        mol.generate_coulomb_matrix(size=29)
    for mol in tqdm(mols_test):
        mol.generate_coulomb_matrix(size=29)

    random.seed(667)

    sigmas = [104857.6]

    X = np.asarray([mol.representation for mol in mols])
    X_test = np.asarray([mol.representation for mol in mols_test])

    Yprime = np.asarray([mol.properties for mol in mols])
    Ytest = np.asarray([mol.properties for mol in mols_test])

    Yprime = np.asarray([i / (621) for i in Yprime])
    Ytest = np.asarray([i / (621) for i in Ytest])

    print("\n -> generate Kernel")

    sigmas = [2200]
    K = laplacian_kernel(X, X, sigmas)
    K_test = laplacian_kernel(X_test, X, sigmas)

    # TODO: Load data into nn and run nn

    hyperparam  = get_config(run_id=args.run)

    model = CoulombNet(10000, layers=hyperparam[8], dropout=hyperparam[3])

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
    list_losses = []
    for e in trange(epochs):
        ids = np.random.shuffle(np.arange(K.shape[0]))
        X_ = K[ids].squeeze()
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
            xt_in = torch.tensor(K_test, dtype=torch.float)
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
            #print("Kenneth mae:", mae)
            print("MAE:", mae_loss)
            print("RMSE:", rmse)
            print(" ===========================")
        list_losses.append(mae_loss)
    print(mean_epoch_loss)
    import csv

    with open('runs/cm_bs_'+str(args.run), 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(list_losses)
