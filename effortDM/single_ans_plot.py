import numpy as np
import inputs as inp
import matplotlib.pyplot as plt
import os
import sys
from matplotlib import cm

Color = {'3x3_1': ['b','orange'],
         '3x3_2': ['k','gray'],
         'q0_1':  ['r','y'],
         'q0_2':  ['purple','k'],
         'cb1':  ['m','g']}
N = '13'
S = '05'
ans = sys.argv[1]
phi = "{:3.2f}".format(float(sys.argv[2])).replace('.','')
dirname = '../Data/phi'+phi+'/'+N+'/'; title = 'With DM interactions'
D = {}
Ji = -0.3
Jf = 0.3
J2 = np.linspace(Ji,Jf,9)
J3 = np.linspace(Ji,Jf,9)
X,Y = np.meshgrid(J2,J3)
Head = inp.header[ans][3:]
head = []
for h in Head:
    head.append(h)
for h in head:
    D[h] = np.zeros((9,9))
for filename in os.listdir(dirname):
    with open(dirname+filename, 'r') as f:
        lines = f.readlines()
    N = (len(lines)-1)//2 + 1
    tempE = []
    for i in range(N):
        data = lines[i*2+1].split(',')
        if data[0] == ans:
            j2 = float(data[1]) - Ji
            j3 = float(data[2]) - Ji
            i2 = int(j2*8/(0.6))
            i3 = int(j3*8/(0.6))
            for n,h in enumerate(head):
                N = n
                if N != 0:
                    try:
                        D[h][i2,i3] = float(data[N+3])
                        if D[h][i2,i3] == 0:
                            D[h][i2,i3] = np.nan
                    except:
                        print("not good: ",h,i2,i3)
                else:
                    D[h][i2,i3] = 1 if data[N+3]=='True' else np.nan
print("Non converged points: ",int(81-D['Converge'].ravel().sum()),"\n",D['Converge'])
nP = len(head)
for i in range(nP):
    temp = []
    for l in range(9):
        for j in range(9):
            if D[head[i]][l,j] == 0:
                D[head[i]][l,j] = np.nan
    for p in D[head[i]][~np.isnan(D[head[i]])].ravel():
        if p != 0 and p != np.nan and p != 'nan':
            temp.append(p)
    print("Range of ",head[i],":",np.amin(temp),"--",np.amax(temp))
    #print("Range of ",head[i],":",np.amin(D[head[i]][np.nonzero(~np.isnan(D[head[i]]))]),"--",np.amax(D[head[i]][~np.isnan(D[head[i]])]))
fig = plt.figure()#(figsize=(16,16))
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.axis('off')
for i in range(nP):
    ax = fig.add_subplot(4,4,i+1,projection='3d')
    ax.plot_surface(X,Y,D[head[i]].T,cmap=cm.coolwarm)
    ax.set_title(ans)
    ax.set_xlabel("J2")
    ax.set_ylabel("J3")
    ax.set_title(head[i])
plt.show()
