import numpy as np
from clust3AB import inputs as inp

inputFile = 'Data/noDMbig/Data_12/J2_J3=(00000_00000).csv'

with open(inputFile, 'r') as f:
    lines = f.readlines()

header = inp.header
D = {}
for i in range(len(lines)//4):
    d = lines[i*4-3].split(',')
    ans = d[0]
    D[ans] = {'ans':ans}
    for p in range(1,len(header[ans])):
        D[ans][header[ans][p]] = float(d[p])


