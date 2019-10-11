import numpy as np
from Bio.PDB import PDBParser
from tqdm import tqdm

PATH = 'data/'

def load_pdb(path=PATH, name = '2L7B'):
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
    return xyz, types

xyz, types = load_pdb()
xyz = np.asarray(xyz)

'''
print(xyz[0])
dist = np.linalg.norm(xyz[0]-xyz[1])
print(dist)
'''

def hist_r(xyz, types, radius = 5):
    regions = []
    for i, atom in enumerate(tqdm(xyz)):
        region = []
        for j in range(xyz.shape[0]):
            if i==j:
                pass
            else:
                dist = np.linalg.norm(atom)
                if dist < radius:
                    region.append([xyz[j], types[j]])
        regions.append(region)
    return regions

regions = hist_r(xyz, types)
print(len(regions))
for l in regions:
    print(len(l))