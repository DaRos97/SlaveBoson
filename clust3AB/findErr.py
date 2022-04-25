import numpy as np
import inputs as inp
import os

dirname = '../Data/yesDMsmall/Data_21/'
j = []
for i in range(1):
    for filename in os.listdir(dirname):
        with open(dirname+filename, 'r') as f:
            lines = f.readlines()
        N = (len(lines)-1)//4 + 1
        for n in range(N):
            data = lines[n*4+1].split(',')
            if data[0] == '3x3':
                j.append([])
                j[-1].append(float(data[1]))
                j[-1].append(float(data[2]))
        continue
        if np.abs(float(data[6])) < 0.51 and np.abs(float(data[6])) > 0.5:
            print("Min at: \tans=",data[0],"\t(J2,J3)=","{:5.4f}".format(float(data[1])),"{:5.4f}".format(float(data[2])))
J = inp.J
li = []
for i1 in range(len(J)):
    a = False
    for i2 in range(len(j)):
        if np.abs(J[i1][0]-j[i2][0]) < inp.cutoff_pts and np.abs(J[i1][1]-j[i2][1]) < inp.cutoff_pts:
            a = True
    if not a:
        li.append(i1)

print(len(li))
