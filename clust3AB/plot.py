import numpy as np
import inputs as inp
import sys
import os

dirname = 'Data/'
E = []
Nans = len(inp.text_ans)
for i,ans in enumerate(['3x3']):#inp.text_ans:
    Dir = dirname+ans+'/'
    dirList = os.listdir(Dir)
    E.append([])
    for file in os.listdir(Dir):
        with open(Dir+file, 'r') as f:
            lines = f.readlines()
        a = lines[1].split(',')
        res = []
        for txt in a:
            res.append(float(txt))
        E[i].append(res[0],res[1],res[2])
    #order the J2,J3 points

minE = []
ordE = np.
for i in range(len(E[0])):
    j2,j3 = E[0][:2]
    for j in range(len(E[0])):
        for l in range(1,Nans):
            d2 = np.abs(j2-E[l][0])

    minE.append(np.argmin(temp))

print(minE)
