import numpy as np
from tqdm import tqdm, trange

PATH = 'data/'

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
    for i in range(len(numberN)):
        print(name + '_' + str(number[i]) + '_' + str(residue[i]) + '.xyz', n_shift[i], h_shift[i])
    return n_shift, h_shift, numberH, numberN

example = '2L7B'
get_shift(name=example)
#n_shift, h_shift, i = get_shift(name=example)