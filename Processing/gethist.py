import numpy as np
from tqdm import tqdm, trange
import glob
import os
import csv
from collections import namedtuple
from Bio.PDB import PDBParser
from sklearn.preprocessing import scale

PATH = '../data/'


def load_pdb(name, path=PATH):
    """
    Load Pdb and get for each atom xyz-coordinates, Atom type (CA, H, N, NZ etc.) and Residue type (ALA, PRO etc.)
    """
    p = PDBParser()
    structure = p.get_structure(name, path + 'raw/' + name + ".pdb")
    types = []
    xyz = []
    res = []
    model = structure[0]
    for chain in model:
        for residue in chain:
            res.append([residue.get_resname(), str(residue.get_id()[1]).zfill(3)])
            for atom in residue:
                xyz.append(atom.get_coord())
                types.append(atom.get_id())
    return xyz, types, res


def make_dist_file(xyz, types, res_list, name, radius):
    """
    Make Histogram for each amino acid.
    Center: N
    Radius: dist = 6
    """
    if not os.path.isdir(PATH + "hist/"):
        os.mkdir(PATH + "hist/")
    hist_list = []
    for i in trange(0, len(xyz)):
        hist = []
        if types[i] == 'N':
            for j in range(0, len(xyz)):
                dist = np.linalg.norm(xyz[i] - xyz[j])
                if dist < radius:
                    # print([types[j], (xyz[j]-xyz[i]).tolist()])
                    hist.append([types[j] + ' ', (xyz[j] - xyz[i]).tolist()])
                else:
                    pass
        if len(hist) > 0:
            hist_list.append(hist)
    for res, hist in zip(res_list, hist_list):
        filename = PATH + 'hist/' + name + '_' + res[1] + '_' + res[0] + '.xyz'
        with open(filename, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerows(hist)


def clean_XYZ(path=PATH):
    """
    clean xyz files with coordinates
    """
    filenames = glob.glob(path + 'hist/*.xyz')
    for filename in tqdm(filenames):
        with open(filename, 'r') as file:
            temp = file.read().replace("[", "   ").replace("]", "   ").replace("HA ", "H  ").replace("HA3 ",
                                                                                                     "H ").replace(
                "NZ ",
                "N  ").replace(
                "HB ", "H  ").replace("HG ", "H  ").replace("HB1 ", "H  ").replace("HB2 ", "H  ").replace("HB3 ",
                                                                                                          "H  ").replace(
                "HG11 ", "H  ").replace("HG12 ", "H  ").replace("HG13 ", "H  ").replace("HG21 ", "H  ").replace("HG22 ",
                                                                                                                "H ").replace(
                "HG23 ", "H  ").replace("HG3 ", "H  ").replace("HG2 ", "H  ").replace("HG1 ", "H  ").replace("HE ",
                                                                                                             "H  ").replace(
                "HE11 ", "H  ").replace("HE12 ", "H  ").replace("HE13 ", "H  ").replace("HE21 ", "H  ").replace("HE22 ",
                                                                                                                "H  ").replace(
                "HE23 ", "H  ").replace("HE3 ", "H  ").replace("HE2 ", "H  ").replace("HE1 ", "H  ").replace("HZ1 ",
                                                                                                             "H  ").replace(
                "HZ2 ", "H  ").replace("HZ3 ", "H  ").replace("HH ", "H  ").replace("HH11 ", "H  ").replace("HH12 ",
                                                                                                            "H  ").replace(
                "HH13 ", "H  ").replace("HH21 ", "H  ").replace("HH22 ", "H  ").replace("HH23 ", "H  ").replace("CA ",
                                                                                                                "C  ").replace(
                "CB ", "C  ").replace("CG ", "C  ").replace("CG1 ", "C  ").replace("CG2 ", "C  ").replace("CE ",
                                                                                                          "C  ").replace(
                "CE1 ", "C  ").replace("CE2 ", "C  ").replace("CZ ", "C  ").replace("CZ2 ", "C  ").replace("CZ3 ",
                                                                                                           "C  ").replace(
                "CH ", "C  ").replace("CH2 ", "C  ").replace("NH1 ", "N  ").replace("NH2 ", "N  ").replace("NE1 ",
                                                                                                           "N  ").replace(
                "NE2 ", "N  ").replace("NE ", "N  ").replace("SG ", "S  ").replace("OG ", "O  ").replace("OG1 ",
                                                                                                         "O  ").replace(
                "OE1 ", "O  ").replace("OE2 ", "O  ").replace("OH ", "O  ").replace("HD11 ", "H  ").replace("HD12 ",
                                                                                                            "H ").replace(
                "HD13 ", "H  ").replace("HD21 ", "H   ").replace("HD22 ", "H  ").replace("HD23 ", "H  ").replace("HD3 ",
                                                                                                                 "H  ").replace(
                "HD2 ", "H  ").replace("HG1 ", "H  ").replace("HD1 ", "H  ").replace("OD1 ", "O  ").replace("OD2 ",
                                                                                                            "O  ").replace(
                "CD ", "C  ").replace("SD ", "S  ").replace("CD1 ", "C  ").replace("CD2 ", "C  ").replace("CE3 ",
                                                                                                          "C ").replace(
                "\"", "   ").replace(",", "   ").replace("HA1 ", "H  ").replace("HA2 ", "H  ").replace("CD1 ",
                                                                                                       "C ").replace(
                "CD1 ", "C  ").replace("ND1 ", "N  ").replace("ND2 ", "N  ").replace("NZ ", "N  ").replace("HZ  ",
                                                                                                           "H   ").replace(
                "H1 ", "H  ").replace("H2 ", "H  ").replace("H3 ", "H   ").replace("HH2 ", "H   ").replace("OXT  ",
                                                                                                           "O   ").replace(
                "ZN ", "N  ").replace("NH ", "N").replace("C1 ", "C").replace("O1 ", "O  ").replace("H21 ",
                                                                                                    "H  ").replace(
                "H22 ", "H  ").replace("H23 ",
                                       "H  ").replace(
                "H41 ", "H  ").replace("H42 ", "H  ").replace("H43 ", "H  ").replace("C2 ", "C  ").replace("C3 ",
                                                                                                           "C  ").replace(
                "C4 ",
                "C  ").replace(
                "H31 ", "H  ").replace("H32 ", "H  ").replace("H33 ", "H  ").replace("H51 ", "H  ").replace("H52 ",
                                                                                                            "H  ").replace(
                "H53 ",
                "H  ").replace(
                "C14 ", "C  ").replace("H141 ", "H  ").replace("H142 ", "H  ").replace("H143 ", "H  ").replace("HBA ",
                                                                                                               "H  ").replace(
                "CX2 ", "C  ").replace("CX3 ",
                                       "C  ").replace(
                "CX1 ", "C   ").replace("O2 ", "O  ").replace("H1A ", "H  ").replace("H1B ", "H  ").replace("H1C ",
                                                                                                            "H  ").replace(
                "H1 ", "H  ").replace("H2 ", "H  ").replace("H3 ", "H  ").replace("H4 ", "H  ").replace("H5 ",
                                                                                                        "H  ").replace(
                "H6 ", "H  ").replace("H7 ", "H  ").replace("H8 ", "H  ").replace("H9 ", "H  ").replace(
                "H10 ", "H  ").replace("C1 ", "C  ").replace("C2 ", "C  ").replace("C3 ", "C  ").replace("C4 ",
                                                                                                         "C  ").replace(
                "C5 ", "C  ").replace("C6 ", "C  ").replace("C7 ", "C  ").replace("C8 ", "C  ").replace("C9 ",
                                                                                                        "C  ").replace(
                "C10 ", "C  ").replace("N1 ", "N  ").replace("N2 ", "N  ").replace("N3 ", "N  ").replace("N4 ",
                                                                                                         "N  ").replace(
                "N5 ", "N  ").replace("N6 ", "N  ").replace("N7 ", "N  ").replace("N8 ", "N  ").replace("N9 ",
                                                                                                        "N  ").replace(
                "N10 ", "N  ").replace("O1 ", "O  ").replace("O2 ", "O  ").replace("O3 ", "O  ").replace("O4 ",
                                                                                                         "O  ").replace(
                "O5 ", "O  ").replace("O6 ", "O  ").replace("O7 ", "O  ").replace("O8 ", "O  ").replace("O9 ",
                                                                                                        "O  ").replace(
                "O10 ", "O  ")
            file.close()
        with open(filename, 'w') as file:
            file.write(temp)
            file.close()

def mv_Res_without_Shift(mode='hist', path=PATH):
    """
    Move all xyz files without NMR shift into a different folder called noshift/ .
    """
    files = glob.glob('data/shift/*.txt')
    fname = []
    print("Finding residues without shift.")
    for file in tqdm(files):
        f = open(file)
        stringList = f.readlines()
        f.close()
        for line in stringList:
            tokens = line.split()
            fname.append(path + mode + '/' + tokens[0])

    print("Moving files with residues without shift.")
    xyzs = glob.glob(path + mode + '/*.xyz')
    for i in tqdm(xyzs):
        if i not in fname:
            os.remove(i)
            # shutil.move(i, 'data/hist_noshift')


def get_all_stupid_atoms(mode='hist', path=PATH):
    fnames = glob.glob(path + mode + '/*.xyz')
    for f in tqdm(fnames):
        i = open(f)
        stringlist = i.readlines()
        i.close()

        for line in stringlist:
            tokens = line.split()
            if len(tokens) >= 2:
                if (tokens[0] != "H" and tokens[0] != "O" and tokens[0] != "N" and tokens[0] != "C" and tokens[0] != "S"):
                    # if (tokens[0] != "1" and tokens[0] != "8" and tokens[0] != "7" and tokens[0] != "6" and tokens[0] != "16"):
                    print(f)
                    try:
                        os.remove(f)
                    except:
                        pass


def get_pH(name, path=PATH):
    files = [path + 'raw/' + name + '.pdb']
    for fname in files:
        f = open(fname)
        stringlist = f.readlines()
        f.close()
        for line in stringlist:
            if "PH " in line:
                a = line.split()
                b = a[-1]
                # print("pH:", b)

        for line in stringlist:
            if "(KELVIN)" in line:
                c = line.split()
                d = c[-1]
                # print("Kelvin:", d)

        for line in stringlist:
            if "IONIC STRENGTH " in line:
                e = line.split()
                f = e[-1]
                # print("Ionic strenght:", f)
    return b, d, f


def addlineto_xyz(name, mode='hist', path=PATH):
    """
    Shape your XYZ
    Get all xyzs in one foleder and txtfile with all shifts and then execute the two functions to get proper xyz file
    """
    file = open(path + 'shift/' + name + '.txt')
    stringList = file.readlines()
    file.close()
    fname = []
    lines = []
    for line in stringList:
        lines.append(line)
        tokens = line.split()
        fname.append(tokens[0])
    xyzs = glob.glob(path + mode + '/*.xyz')

    b, d, f = get_pH(name=name)

    with open(path + 'shift/' + name + '.txt') as openfile:
        for line in openfile:
            for part in line.split():
                for j in xyzs:
                    if j in path + mode + '/' + part:
                        filey = open(j)
                        stringListy = filey.readlines()
                        filey.close()
                        stringListy = [l.strip('\n').strip(' ') for l in stringListy]
                        with open(j, "w") as outfile:
                            outfile.write(str(len(stringListy)) + ' ' + b + ' ' + d + ' ' + f + ' ' + line)
                            outfile.writelines("%s\n" % line for line in stringListy)
                        outfile.close()
    openfile.close()
    return


i = ['2L7B', '2L7Q']

xyz, types, res = load_pdb(path=PATH, name=i)
xyz = np.asarray(xyz)
res = np.asarray(res)
make_dist_file(xyz, types, res, name=i, radius=6)
clean_XYZ()
clean_XYZ()
get_all_stupid_atoms()

limit = np.inf
new_src_path = PATH + 'raw/*.pdb'

filenames = glob.glob(pathname=new_src_path)
print(len(filenames))

if limit < len(filenames):
    filenames = filename[:limit]
print("Starting to preprocess {} proteins...".format(len(filenames)))

id_list = list(name[12:16] for name in filenames)

for i in tqdm(id_list):
    addlineto_xyz(i, mode='hist')
    #addlineto_xyz(i, mode='CV')

mv_Res_without_Shift()



"""xyz, types, res = load_pdb(path=PATH, name='2L7B')
xyz = np.asarray(xyz)
res = np.asarray(res)
make_dist_file(xyz, types, res, name='2L7B', radius=6)
clean_XYZ()
clean_XYZ()
get_all_stupid_atoms()
addlineto_xyz(name='2L7B')"""

