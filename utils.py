from configparser import ConfigParser
import argparse
import ast
import os


def get_config(run_id = 1, path = "config/config_"):
    config_file = path + str(run_id).zfill(5) + ".ini"
    print("Loading config file.. " + config_file)
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


def get_config2(run_id = 1, path = "config/config_"):
    config_file = path + str(run_id).zfill(5) + ".ini"
    print("Loading config file.. " + config_file)
    parser = ConfigParser()
    parser.read(config_file)
    model = int(parser['SETTING']['model'])
    lr = float(parser['SETTING']['lr'])
    epochs = int(parser['SETTING']['epochs'])
    batchsize = int(parser['SETTING']['batchsize'])
    neighborradius = float(parser['SETTING']['neighborradius'])
    nclouds = int(parser['SETTING']['nclouds'])
    clouddim = int(parser['SETTING']['clouddim'])
    resblocks = int(parser['SETTING']['resblocks'])
    nffl = int(parser['SETTING']['nffl'])
    ffl1size = int(parser['SETTING']['ffl1size'])
    emb_dim = int(parser['SETTING']['emb_dim'])
    cloudord = int(parser['SETTING']['cloudord'])
    return model, lr, neighborradius, nclouds, clouddim, resblocks, epochs, batchsize, nffl, ffl1size, emb_dim, cloudord