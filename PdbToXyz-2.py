"""
This script transfors pdb into xyz and creates for every pdb a new xyz

    Pdb format:
    Atom, number, atom, chain A, residue number, x,y,z, occupancy=1.00, Temp factor, atom only

    XYZ: Format:
    Number of Atoms
    File Name
    Atom    X   Y  Z
    ...

    For other files: just bash is enough:
    babel *.smi -oxyz -m --gen3d
"""

import numpy as np
import glob
import os

filenames = glob.glob('*.pdb')

for filename in filenames:
    f = open(filename, "r")
    lines = f.readlines()
    f.close()

X_data = []
Y_data = []
Z_data = []
atom = []

for line in lines:
    tokens = line.split()
    if len(tokens) == 12:
        X_data.append(float(tokens[6]))
        Y_data.append(float(tokens[7]))
        Z_data.append(float(tokens[8]))
        atom.append(tokens[11])
        xyz = list(zip(atom, X_data ,Y_data ,Z_data ))

for filename in filenames:
    with open(filename + '.xyz', 'w') as outfile:
        print(len(xyz), file=outfile)
        print(os.path.splitext(filename)[0], file=outfile)
        for i in xyz:
            print(i, file=outfile)


""""
after creatimng files rename files in bash:

for file in *.xyz; do mv "$file" "${file/.pdb.xyz/.xyz}"; done

"""
