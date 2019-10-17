import numpy as np
from Bio.PDB import PDBParser
from tqdm import tqdm, trange
import csv
import os

PATH = 'data/'

def load_pdb(name, path=PATH):
    p = PDBParser()
    structure = p.get_structure(name, path+name+"_pdb.txt")
    types = []
    xyz = []
    res = []
    model = structure[0]
    counter = 1
    for chain in model:
        for residue in chain:
            res.append([residue.get_resname(), "%03d" % counter])
            for atom in residue:
                xyz.append(atom.get_coord())
                types.append(atom.get_id())
            counter += 1

    return xyz, types, res

def make_dist_file(name, xyz, types, res_list):
    hist_list = []
    for i in trange(0, len(xyz)):
        hist = []
        if types[i] == 'N':
            for j in range(0, len(xyz)):
                if i >= 1:
                    dist = np.linalg.norm(xyz[i]-xyz[j])
                    if dist < 6:
                        hist.append([types[j], xyz[j].tolist()])
                else:
                    pass
        if len(hist) > 0:
            hist_list.append(hist)
    for res, hist in zip(res_list, hist_list):
        filename = PATH + name + '_' + res[1] + '_' + res[0] + '.txt'
        with open(filename, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerows(hist)

example = '2L7B'
xyz, types, res = load_pdb(name=example)
xyz = np.asarray(xyz)
res = np.asarray(res)
make_dist_file(example, xyz, types, res)
