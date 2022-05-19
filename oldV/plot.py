import numpy as np
import inputs as inp
import matplotlib.pyplot as plt
import os
import sys

Color = {'3x3': ['b','orange'],
         'q0':  ['r','y'],
         '0-pi': 'y',
         'cb1':  ['m','g'],
         'cb2': 'k'}
N = int(sys.argv[1])
dirname = '../Data/noDMbig/Data_'+sys.argv[1]+'/'
#dirname = '../Data/yesDMbig/Data_'+sys.argv[1]+'/'
#dirname = '../Data/noDMsmall/copy/'#Data_'+sys.argv[1]+'/'
#dirname = '../Data/yesDMsmall/copy/'#Data_'+sys.argv[1]+'/'
#dirname = '../Data/yesDMsmall/Data_'+sys.argv[1]+'N/'
minE = []
ct = 0
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
    minE.append(lines[minInd*4+1].split(','))

pts = len(os.listdir(dirname))
#check on convergence
plt.figure(figsize=(16,16))
plt.subplot(2,3,2)
for p in range(pts):
    J2 = float(minE[p][1])
    J3 = float(minE[p][2])
    conv = '^'
    if float(minE[p][4]) < 1e-8:# and float(minE[p][6]) > 0.5:
        conv = 'o'
    if float(minE[p][5]) < ct:
        col = Color[minE[p][0]][0]
    elif float(minE[p][5]) > ct:
        col = Color[minE[p][0]][1]
    plt.scatter(float(minE[p][1]),float(minE[p][2]),color=col,marker = conv)
plt.hlines(0,inp.J2i,inp.J2f,'g',linestyles = 'dashed')
plt.vlines(0,inp.J3i,inp.J3f,'g',linestyles = 'dashed')

for ind,i in enumerate(['3x3', 'q0', 'cb1']):
    plt.subplot(2,3,ind+4)
    plt.title(i)
    for p in range(len(E[i])):
        conv='^'
        J2 = float(E[i][p][1])
        J3 = float(E[i][p][2])
        try:
            if float(E[i][p][4]) < 1e-8:
                conv = 'o'
            if float(E[i][p][5]) < ct:
                col = Color[E[i][p][0]][0]
            elif float(E[i][p][5]) > ct:
                col = Color[E[i][p][0]][1]
            plt.scatter(J2,J3,color=col,marker = conv)
        except:
            print(J2,J3,i," did not")
#real fig

plt.show()
