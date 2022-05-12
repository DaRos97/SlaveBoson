import numpy as np
import inputs as inp
import matplotlib.pyplot as plt
import os
import sys
from matplotlib import cm

Color = {'3x3': ['b','orange'],
         'q0':  ['r','y'],
         '0-pi': 'y',
         'cb1':  ['m','g'],
         'cb2': 'k'}
N = str(int(sys.argv[1]))
#dirname = '../Data/yesDMbig/Data_'+N+'-'+N+'/'; title = 'With DM interactions'
dirname = '../Data/noDMbig/Data_'+N+'-'+N+'/'; title = 'Without DM interactions'
#dirname = '../Data/yesDMbig/Data_'+sys.argv[1]+'/'
#dirname = '../Data/noDMbig/Data_'+sys.argv[1]+'/'
#dirname = '../Data/yesDMbig/Data_'+sys.argv[1]+'N/'
minE = []
ct = {'3x3':1e-2, 'q0':1e-2,'cb1':1e-2}
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
        data = lines[i*4+1].split(',')
        tE = [data[0]]
        tempE.append(float(data[3]))     #ans,J2,J3,E,S
        for j in range(1,len(data)):
            tE.append(float(data[j]))
        E[data[0]].append(tE)
    minInd = np.argmin(np.array(tempE))
    minE.append(lines[minInd*4+1].split(','))
cn = int(input("Compute gaps(0), energies(1) or both(2)?"))
list_ans = E.keys()
if cn == 0 or cn == 2:
    gaps = {}
    Ji = -0.3
    Jf = 0.3
    fig = plt.figure(figsize=(16,16))
    tt = title + ': gap values'
    plt.title(tt)
    plt.axis('off')
    for a,ans in enumerate(list_ans):
        gaps[ans] = np.zeros((9,9))
        for j2 in range(len(E[ans])):
            J2 = E[ans][j2][1] - Ji
            i2 = int(J2*8/(0.6))
            J3 = E[ans][j2][2] - Ji
            i3 = int(J3*8/(0.6))
            gaps[ans][i2,i3] = E[ans][j2][5]
        J2 = np.linspace(Ji,Jf,9)
        J3 = np.linspace(Ji,Jf,9)
        X,Y = np.meshgrid(J2,J3)
        Z = np.sin(np.sqrt(X**2+Y**2))
        n = int(100 + len(list_ans)*10 +a+1)
        ax = fig.add_subplot(n,projection='3d')
        ax.plot_surface(X,Y,gaps[ans].T,cmap=cm.coolwarm)
        ax.set_title(ans)
        ax.set_xlabel("J2")
        ax.set_ylabel("J3")
    plt.show()
if cn == 0:
    exit()
pts = len(os.listdir(dirname))
#check on convergence
plt.figure(figsize=(16,16))
plt.subplot(2,3,2)
tt = title + ': energy'
plt.title(tt)
for p in range(pts):
    J2 = float(minE[p][1])
    J3 = float(minE[p][2])
    conv = '^'
    if float(minE[p][4]) < 1e-8:# and float(minE[p][6]) > 0.5:
        conv = 'o'
    if float(minE[p][5]) < ct[minE[p][0]]:
        col = Color[minE[p][0]][0]
    else:
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
            if float(E[i][p][5]) < ct[E[i][p][0]]:
                col = Color[E[i][p][0]][0]
            else:
                col = Color[E[i][p][0]][1]
            plt.scatter(J2,J3,color=col,marker = conv)
        except:
            print(J2,J3,i," did not")
#Legenda
plt.subplot(2,3,3)
plt.axis('off')
for a,ans in enumerate(list_ans):
    txt = ans + ': LRO -> '+Color[ans][0]+', SL -> '+Color[ans][1]
    plt.text(0.1,0.1+a/5,txt)


plt.show()
