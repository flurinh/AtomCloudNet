import glob

filenames = glob.glob('*_pdb.txt')

for filename in filenames:
    with open(filename) as infile, open(filename + '.pdb', 'w') as outfile:
        copy = False
        for line in infile:
            if line.strip() == "MODEL        1":
                copy = True
                continue
            elif line.strip() == "MODEL        2":
                copy = False
                continue
            elif copy:
                outfile.write(line)

""""
after spliting files rename files in bash:

for file in *.pdb; do mv "$file" "${file/_pdb.txt.pdb/.pdb}"; done

"""
