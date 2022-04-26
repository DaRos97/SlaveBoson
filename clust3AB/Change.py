import inputs as inp
import numpy as np
import os
import common_functions as cf

dir1 = '../temp/'
dir2 = '../Data/noDMsmall/Data_21/'

for f1 in os.listdir(dir1):
    with open(dir1+f1, 'r') as f:
        lin1 = f.readlines()
    d1a = lin1[16].split(',')
    d11 = ['cb1']
    for i in d1a[1:-1]:
        d11.append(float(i.split(': ')[1]))
    d11.append(float(d1a[-1].split(': ')[1][:-2]))
    
    d1b = lin1[17].split(',')
    d12 = []
    for i in d1b[:-1]:
        d12.append(float(i.split(': ')[1]))
    d12.append(float(d1b[-1].split(': ')[1][:-2]))

    J2 = d11[1]
    J3 = d11[2]
    header = inp.header['cb1']
    Data = {}
    Hess = {}
    for i in range(len(d12)):
        Hess[header[6+i]] = d12[i]
    for ind in range(len(d11)):
        Data[header[ind]] = d11[ind]
    for f2 in os.listdir(dir2):
        with open(dir2+f2, 'r') as f:
            lin2 = f.readlines()
        N = (len(lin2)-1)//4 + 1
        for n in range(N):
            data = lin2[n*4+1].split(',')
            if data[0] == 'cb1' and np.abs(float(data[1])-J2) < inp.cutoff and  np.abs(float(data[2])-J3) < inp.cutoff:
                cf.SaveToCsv(Data,Hess,dir2+f2)


