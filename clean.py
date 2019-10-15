
import numpy as np
import glob

filenames = glob.glob('*txt')



for filename in filenames:
    with open(filename) as infile, open(filename + '.txt', 'w') as outfile:
        temp = infile.read().replace("[", "  ").replace("]", "  ").replace("HA ", "H ").replace("HB ", "H ").replace("HG ", "H ").replace("HB1 ", "H ").replace("HB2 ", "H ").replace("HB3 ", "H ").replace("HG11 ", "H ").replace("HG12 ", "H ").replace("HG13 ", "H ").replace("HG21 ", "H ").replace("HG22 ", "H ").replace("HG23 ", "H ").replace("HG3 ", "H ").replace("HG2 ", "H ").replace("HG1 ", "H ").replace("HE ", "H ").replace("HE11 ", "H ").replace("HE12 ", "H ").replace("HE13 ", "H ").replace("HE21 ", "H ").replace("HE22 ", "H ").replace("HE23 ", "H ").replace("HE3 ", "H ").replace("HE2 ", "H ").replace("HE1 ", "H ").replace("HZ1 ", "H ").replace("HZ2 ", "H ").replace("HZ3 ", "H ").replace("HH ", "H ").replace("HH11 ", "H ").replace("HH12 ", "H ").replace("HH13 ", "H ").replace("HH21 ", "H ").replace("HH22 ", "H ").replace("HH23 ", "H ").replace("CA ", "C ").replace("CB ", "C ").replace("CG ", "C ").replace("CG1 ", "C ").replace("CG2 ", "C ").replace("CE ", "C ").replace("CE1 ", "C ").replace("CE2 ", "C ").replace("CZ ", "C ").replace("CZ2 ", "C ").replace("CZ3 ", "C ").replace("CH ", "C ").replace("CH2 ", "C ").replace("NH1 ", "N").replace("NH2 ", "N ").replace("NE1 ", "N ").replace("NE2 ", "N ").replace("NE ", "N ").replace("SG ", "S ").replace("OG ", "O ").replace("OG1 ", "O ").replace("OE1 ", "O ").replace("OE2 ", "O ").replace("OH ", "O ").replace("HD11 ", "H ").replace("HD12 ", "H ").replace("HD13 ", "H ").replace("HD21 ", "H ").replace("HD22 ", "H ").replace("HD23 ", "H ").replace("HD3 ", "H ").replace("HD2 ", "H").replace("HG1 ", "H ").replace("HD1 ", "H ").replace("OD1 ", "O ").replace("OD2 ", "O ").replace("CD ", "C ").replace("SD ", "S ").replace("CD1 ", "C ").replace("CD2 ", "C ").replace("CE3 ", "C ").replace("\"", " ").replace(",", " ").replace("HA1 ", "H ").replace("HA2 ", "H ")
        print(temp, file=outfile)


""""
    after creatimng files rename files in bash:
    for file in *.txt; do mv "$file" "${file/.txt.txt/.txt}"; done
    
    """
