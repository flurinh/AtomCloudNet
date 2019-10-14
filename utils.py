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
    model = structure[0]
    for chain in model:
        for residue in chain:
            for atom in residue:
                xyz.append(atom.get_coord())
                types.append(atom.get_id())
    return xyz, types, res

def make_dist_file(name, xyz, types):
    for i in trange(0, len(xyz)):
        hist = []
        for j in range(0, len(xyz)):
            if types[i] == 'N':
                dist = np.linalg.norm(xyz[i]-xyz[j])
                if dist < 6:
                    hist.append([types[j], xyz[j].tolist()])
                else:
                    pass
        filename = PATH+name+'/'+ name + '_' + str(i)
        if not os.path.isdir(PATH+name):
            os.mkdir(PATH+name)
        if types[i] == 'N':
            if len(hist) > 0:
                with open(filename, 'w', newline='') as f:
                    wr = csv.writer(f)
                    wr.writerows(hist)


example = '2L7B'
xyz, types, res = load_pdb(name=example)
xyz = np.asarray(xyz)
make_dist_file(example, xyz, types)
