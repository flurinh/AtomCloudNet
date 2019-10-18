import glob
import shutil


path = "shift/"
example = "2L7B"
name = example
destination = "noshift/"

def mv_Res_without_Shift(name, PATH=path):

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

#mv_Res_without_Shift(name)




fname = glob.glob('*.xyz')

def file_len(fname):
    for f in fname:
        file = open(f)
        stringList = file.readlines()
        if len(stringList) >= 119:
            print(len(stringList))


print(file_len(fname))
