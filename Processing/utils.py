import numpy as np
from Bio.PDB import PDBParser
from tqdm import tqdm, trange
import csv
import shutil
import glob

PATH = 'data/'


"""

Load Pdb and get for each atom xyz-coordinates, Atom type (CA, H, N, NZ etc.) and Residue type (ALA, PRO etc.)

"""


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


"""

Make Histogram for each amino acid.

Center: N

Radius: dist = 6

"""

def make_dist_file(name, xyz, types, res_list):
    hist_list = []
    for i in trange(0, len(xyz)):
        hist = []
        if types[i] == 'N':
            for j in range(0, len(xyz)):
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


"""

clean xyz files with coordinates

"""
def clean_XYZ()
    filenames = glob.glob('*txt')
    for filename in filenames:
        with open(filename) as infile, open(filename + '.txt', 'w') as outfile:
            temp = infile.read().replace("[", "  ").replace("]", "  ").replace("HA ", "H ").replace("HA3 ", "H ").replace("NZ ", "N ").replace("HB ", "H ").replace("HG ", "H ").replace("HB1 ", "H ").replace("HB2 ", "H ").replace("HB3 ", "H ").replace("HG11 ", "H ").replace("HG12 ", "H ").replace("HG13 ", "H ").replace("HG21 ", "H ").replace("HG22 ", "H ").replace("HG23 ", "H ").replace("HG3 ", "H ").replace("HG2 ", "H ").replace("HG1 ", "H ").replace("HE ", "H ").replace("HE11 ", "H ").replace("HE12 ", "H ").replace("HE13 ", "H ").replace("HE21 ", "H ").replace("HE22 ", "H ").replace("HE23 ", "H ").replace("HE3 ", "H ").replace("HE2 ", "H ").replace("HE1 ", "H ").replace("HZ1 ", "H ").replace("HZ2 ", "H ").replace("HZ3 ", "H ").replace("HH ", "H ").replace("HH11 ", "H ").replace("HH12 ", "H ").replace("HH13 ", "H ").replace("HH21 ", "H ").replace("HH22 ", "H ").replace("HH23 ", "H ").replace("CA ", "C ").replace("CB ", "C ").replace("CG ", "C ").replace("CG1 ", "C ").replace("CG2 ", "C ").replace("CE ", "C ").replace("CE1 ", "C ").replace("CE2 ", "C ").replace("CZ ", "C ").replace("CZ2 ", "C ").replace("CZ3 ", "C ").replace("CH ", "C ").replace("CH2 ", "C ").replace("NH1 ", "N").replace("NH2 ", "N ").replace("NE1 ", "N ").replace("NE2 ", "N ").replace("NE ", "N ").replace("SG ", "S ").replace("OG ", "O ").replace("OG1 ", "O ").replace("OE1 ", "O ").replace("OE2 ", "O ").replace("OH ", "O ").replace("HD11 ", "H ").replace("HD12 ", "H ").replace("HD13 ", "H ").replace("HD21 ", "H ").replace("HD22 ", "H ").replace("HD23 ", "H ").replace("HD3 ", "H ").replace("HD2 ", "H").replace("HG1 ", "H ").replace("HD1 ", "H ").replace("OD1 ", "O ").replace("OD2 ", "O ").replace("CD ", "C ").replace("SD ", "S ").replace("CD1 ", "C ").replace("CD2 ", "C ").replace("CE3 ", "C ").replace("\"", " ").replace(",", " ").replace("HA1 ", "H ").replace("HA2 ", "H ").replace("CD1 ", "C ").replace("CD1 ", "C ").replace("ND1 ", "N ").replace("ND2 ", "N ").replace("NZ ", "N ").replace("HZ  ", "H  ").replace("H1  ", "H  ").replace("H2  ", "H  ").replace("H3  ", "H  ").replace("HH2  ", "H  ").replace("OXT  ", "O  ").replace("ZN  ", "N  ")
            print(temp, file=outfile)




"""

Create file txt file with filename and chemical shift:

*.xyz   H-Shift

"""

def get_shift(name, path=PATH):

    file = open(path + name + ".txt")
    stringList = file.readlines()
    file.close()

    n_shift = []
    h_shift = []
    numberN = []
    numberH = []
    number = []
    residue = []

    for line in stringList:
        tokens = line.split()
        if len(tokens) >= 27:
            if 'N    ' in line:
                n_shift.append(float(tokens[9]))
                numberN.append(float(tokens[0]))
            if 'H    ' in line:
                h_shift.append(float(tokens[9]))
                numberH.append(float(tokens[0]))
                number.append(int(tokens[17]))
                residue.append(tokens[5])

    fname = PATH + "shift/" + name + '.txt'
    with open(fname, 'w') as outfile:
        for i in range(len(numberN)):
            num = ["%.3d" % x for x in number]
            print(name + '_' + str(num[i]) + '_' + str(residue[i]) + '.xyz', n_shift[i], h_shift[i], file=outfile)
    return n_shift, h_shift, numberH, numberN


"""

Move all xyz files without NMR shift into a different folder called noshift/ . 

"""

PATH2 = "shift/"
destination = "noshift/"

def mv_Res_without_Shift(name, path=PATH2):

    file = open(path + name + ".txt")
    stringList = file.readlines()
    file.close()

    fname = []

    for line in stringList:
        tokens = line.split()
        fname.append(tokens[0])

    xyzs = glob.glob('*.xyz')

    for i in xyzs:
        if i in xyzs and i not in fname:
            shutil.move(i, destination)


"""

Shape your XYZ 


Get all xyzs in one foleder and txtfile with all shifts and then execute the two functions to get proper xyz file 

"""

PATH = '../xyzcopy/'

def addlineto_xyz(name, path=PATH):

    file = open(path + name + ".txt")
    stringList = file.readlines()
    file.close()

    fname = []
    lines = []

    for line in stringList:
        lines.append(line)
        tokens = line.split()
        fname.append(tokens[0])

    xyzs = glob.glob('*.xyz')


    with open(path + name + ".txt") as openfile:
        for line in openfile:
            for part in line.split():
                for j in xyzs:
                    if j in part:
                        filey = open(j)
                        stringListy = filey.readlines()
                        with open(j + ".txt", "w") as outfile:
                            print(len(stringListy), file=outfile)
                            print(line, file=outfile)
                            for liney in stringListy:
                                print(liney, file=outfile)



def strip_txt_file():
    filenames = glob.glob('*.xyz.txt')
    for filename in filenames:
        with open(filename) as infile, open(filename + '.txt', 'w') as outfile:
            for line in infile:
                if not line.strip(): continue
                outfile.write(line)



#addlineto_xyz(name='2L7B')
strip_txt_file()


"""

rm *.xyz


for file in *.xyz.txt.txt; do mv "$file" "${file/.xyz.txt.txt/.xyz}"; done

rm *.txt

"""
