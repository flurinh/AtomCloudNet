import numpy as np
import glob

filenames = glob.glob('*xyz')



for filename in filenames:
    with open(filename) as infile, open(filename + '.pdb', 'w') as outfile:
        temp = infile.read().replace("(", "").replace(")", "").replace("'", "").replace(",", "")
        print(temp, file=outfile)


""""
after creatimng files rename files in bash:

for file in *.pdb; do mv "$file" "${file/.xyz.pdb/.xyz}"; done

"""
