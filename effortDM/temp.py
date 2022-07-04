import numpy as np
import inputs as inp
import os
import common_functions as cf

dirname = '/home/users/r/rossid/Data/fullDM_13/'
dirname2 = '/home/users/r/rossid/Data/fullDM_13_2/'
head = inp.header
for filename in os.listdir(dirname):
    with open(dirname+filename, 'r') as f:
        lines = f.readlines()
    N = (len(lines)-1)//2 + 1
    D = {}
    P = {}
    new_filename = dirname2+filename
    for i in range(N):
        data = lines[i*2+1].split(',')
        H = head[data[0]][:3] + head[data[0]][4:]
        D[data[0]] = {}
        P[data[0]] = []
        for n,h in enumerate(H):
            if n == 0:
                D[data[0]][h] = data[n]
            else:
                D[data[0]][h] = float(data[n])
                if n >= 8 and np.abs(float(data[n])) > 1e-13:
                    P[data[0]].append(float(data[n]))
        bnds = cf.FindBounds(float(data[1]),float(data[2]),[data[0]])[data[0]]
        conv = cf.IsConverged(P[data[0]],bnds,float(data[4]))
        D[data[0]]['Converge'] = conv
        cf.SaveToCsv(D[data[0]],new_filename)
