import numpy as np
import inputs as inp
import matplotlib.pyplot as plt
import os
import sys

N = int(sys.argv[1])
dirname = '../Data/Data_'+sys.argv[1]+'N/'
minE = []
E = []
for file in os.listdir(dirname):
    with open(dirname+file, 'r') as f:
        lines = f.readlines()
    N = (len(lines)-1)//4 + 1
    tempE = []
    tE = []
    if N != 3:
        print(file," has only ",N," ansatze.")
    for i in range(N):
        tE.append([])
        data = lines[i*4+1].split(',')
        tempE.append(float(data[3]))     #ans,J2,J3,E,S
        for j in range(5):
            tE[i].append(data[j])
    E.append(tE)
    minInd = np.argmin(np.array(tempE))
    minE.append(lines[minInd*4+1].split(',')[:5])

pts = len(E)
Color = {'3x3': 'b',
         'q0':   'r',
         'cb1':  'm'}
plt.figure(figsize=(16,16))
plt.subplot(2,2,1)
for p in range(pts):
    conv = '^'
    if float(minE[p][4]) < 1e-8:
        conv = 'o'
    plt.scatter(float(minE[p][1]),float(minE[p][2]),color=Color[minE[p][0]],marker = conv)
for i in range(3):
    plt.subplot(2,2,2+i)
    plt.title(E[0][i][0])
    for p in range(pts):
        conv='^'
        if float(E[p][i][4]) < 1e-8:
            conv = 'o'
        plt.scatter(float(E[p][i][1]),float(E[p][i][2]),color=Color[E[p][i][0]],marker = conv)

plt.show()
