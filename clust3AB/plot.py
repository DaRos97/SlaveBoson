import numpy as np
import inputs as inp
import matplotlib.pyplot as plt
import os
import sys

N = int(sys.argv[1])
dirname = '../Data/pdSmall/Data_'+sys.argv[1]+'/'
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
        for j in range(len(data)):
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
#check on convergence
plt.figure(figsize=(16,16))
#plt.subplot(2,3,2)
for p in range(pts):
    conv = '^'
    if float(minE[p][4]) < 1e-8:# and float(minE[p][6]) > 0.5:
        conv = 'o'
    elif float(E[minE[p][0]][p][6]) < 0.5:
        conv = '*'
    plt.scatter(float(minE[p][1]),float(minE[p][2]),color=Color[minE[p][0]],marker = conv)
plt.hlines(0,inp.J2i,inp.J2f,'g',linestyles = 'dashed')
plt.vlines(0,inp.J3i,inp.J3f,'g',linestyles = 'dashed')

plt.show()
exit()
for ind,i in enumerate(['3x3', 'q0', 'cb1']):
    plt.subplot(2,3,ind+4)
    plt.title(i)
    for p in range(len(E[i])):
        conv='^'
        try:
            if float(E[i][p][4]) < 1e-8:
                conv = 'o'
            elif float(E[i][p][6]) < 0.5:
                conv = '*'
            plt.scatter(float(E[i][p][1]),float(E[i][p][2]),color=Color[E[i][p][0]],marker = conv)
        except:
            continue
#real fig

plt.show()
