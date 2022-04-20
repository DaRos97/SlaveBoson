import numpy as np
import inputs as inp
import matplotlib.pyplot as plt
import os
import sys

N = int(sys.argv[1])
dirname = '../Data/Data_'+sys.argv[1]+'/'
minE = []
E = {'3x3':[],
     'q0' :[],
     'cb1':[]
     }
for filename in os.listdir(dirname):
    with open(dirname+filename, 'r') as f:
        lines = f.readlines()
    N = (len(lines)-1)//4 + 1
    tempE = []
    for i in range(N):
        tE = []
        data = lines[i*4+1].split(',')
        tempE.append(float(data[3]))     #ans,J2,J3,E,S
        for j in range(5):
            tE.append(data[j])
        E[data[0]].append(tE)
    minInd = np.argmin(np.array(tempE))
    minE.append(lines[minInd*4+1].split(',')[:5])

pts = len(os.listdir(dirname))
Color = {'3x3': 'b',
         'q0':   'r',
         '0-pi': 'y',
         'cb1':  'm',
         'cb2': 'k'}
plt.figure(figsize=(16,16))
for p in range(pts):
    conv = '^'
    if float(minE[p][4]) < 1e-8:
        conv = 'o'
    plt.scatter(float(minE[p][1]),float(minE[p][2]),color=Color[minE[p][0]],marker = conv)
plt.show()

plt.figure(figsize=(16,16))
for ind,i in enumerate(['3x3', 'q0', 'cb1']):
    plt.subplot(1,3,ind+1)
    plt.title(i)
    for p in range(len(E[i])):
        conv='^'
        try:
            if float(E[i][p][4]) < 1e-8:
                conv = 'o'
            plt.scatter(float(E[i][p][1]),float(E[i][p][2]),color=Color[E[i][p][0]],marker = conv)
        except:
            continue

plt.show()
