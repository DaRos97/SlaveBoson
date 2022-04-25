import numpy as np
import inputs as inp
import matplotlib.pyplot as plt
import os
import sys

N = int(sys.argv[1])
#dirname = '../Data/noDMbig/Data_'+sys.argv[1]+'N/'
dirname = '../Data/noDMsmall/Data_'+sys.argv[1]+'/'
#dirname = '../Data/yesDMsmall/Data_'+sys.argv[1]+'/'
minE = []
E = {'3x3':[],
     'q0' :[],
     'cb1':[]
     }
H = {'3x3':[],
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
        tH = []
        data = lines[i*4+1].split(',')
        data2 = lines[i*4+3].split(',')
        tempE.append(float(data[3]))     #ans,J2,J3,E,S
        for j in range(len(data)):
            tE.append(data[j])
        for j in range(len(data2)):
            tH.append(data2[j])
        E[data[0]].append(tE)
        H[data[0]].append(tH)
    minInd = np.argmin(np.array(tempE))
    minE.append(lines[minInd*4+1].split(','))

pts = len(os.listdir(dirname))
Color = {'3x3': 'b',
         'q0':   'r',
         '0-pi': 'y',
         'cb1':  'm',
         'cb2': 'k'}
#check on convergence
plt.figure(figsize=(16,16))
plt.subplot(2,3,2)
for p in range(pts):
    J2 = float(minE[p][1])
    J3 = float(minE[p][2])
    j2 = int(np.sign(J2)*np.sign(int(np.abs(J2)*1e8)) + 1)   #j < 0 --> 0, j == 0 --> 1, j > 0 --> 2
    j3 = int(np.sign(J3)*np.sign(int(np.abs(J3)*1e8)) + 1)
    conv = '^'
    if float(minE[p][4]) < 1e-8:# and float(minE[p][6]) > 0.5:
        conv = 'o'
    if np.abs(float(minE[p][6])) < 0.5:
        conv = '*'
    cn = 0
    for n in range(len(H[minE[p][0]][p])):
        hess = int(np.sign(float(H[minE[p][0]][p][n])))
        if hess != 0:
            if hess != inp.HS[minE[p][0]][j2][j3][cn]:
                conv = '^'
            cn += 1
    plt.scatter(float(minE[p][1]),float(minE[p][2]),color=Color[minE[p][0]],marker = conv)
plt.hlines(0,inp.J2i,inp.J2f,'g',linestyles = 'dashed')
plt.vlines(0,inp.J3i,inp.J3f,'g',linestyles = 'dashed')

for ind,i in enumerate(['3x3', 'q0', 'cb1']):
    plt.subplot(2,3,ind+4)
    plt.title(i)
    for p in range(len(E[i])):
        conv='^'
        J2 = float(E[i][p][1])
        J3 = float(E[i][p][2])
        j2 = int(np.sign(J2)*np.sign(int(np.abs(J2)*1e8)) + 1)   #j < 0 --> 0, j == 0 --> 1, j > 0 --> 2
        j3 = int(np.sign(J3)*np.sign(int(np.abs(J3)*1e8)) + 1)
        try:
            if float(E[i][p][4]) < 1e-8:
                conv = 'o'
            if np.abs(float(E[i][p][6])) < 0.5:
                conv = '*'
            cn = 0
            for n in range(len(E[i][p][6:])):
                hess = int(np.sign(float(H[i][p][n])))
                if hess != 0:
                    if hess != inp.HS[i][j2][j3][cn]:
                        conv = '^'
                    cn += 1
            plt.scatter(J2,J3,color=Color[E[i][p][0]],marker = conv)
        except:
            print(J2,J3,i)
            print(j2,j3)
            print(inp.HS[i][j2][j3])
            print(len(H[i][p]))
            for n in range(len(H[i][p])):
                hess = int(np.sign(float(H[i][p][n])))
                if hess == 0:
                    continue
                if hess != inp.HS[i][j2][j3][n]:
                    conv = '^'
            continue
#real fig

plt.show()
