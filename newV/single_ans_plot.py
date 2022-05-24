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
         'cb2': ['k','purple']}
#dirname = '../Data/noDM/Data_13-13/'; title = 'Without DM interactions'
dirname = '../Data/yesDM/Data_13-13/'; title = 'With DM interactions'
if len(sys.argv) > 1:
    ans = sys.argv[1]
else:
    ans = input("Which ans?(3x3,q0,cb1)")
D = {}
Ji = -0.3
Jf = 0.3
J2 = np.linspace(Ji,Jf,9)
J3 = np.linspace(Ji,Jf,9)
X,Y = np.meshgrid(J2,J3)
Head = inp.header[ans][3:]
head = []
for h in Head:
    if h != 'Sigma':
        head.append(h)
for h in head:
    D[h] = np.zeros((9,9))
for filename in os.listdir(dirname):
    with open(dirname+filename, 'r') as f:
        lines = f.readlines()
    N = (len(lines)-1)//4 + 1
    tempE = []
    for i in range(N):
        data = lines[i*4+1].split(',')
        if data[0] == ans:
            j2 =float(data[1]) - Ji
            j3 =float(data[2]) - Ji
            i2 = int(j2*8/(0.6))
            i3 = int(j3*8/(0.6))
            for n,h in enumerate(head):
                if n >= 1:
                    N = n+1
                else:
                    N = n
                D[h][i2,i3] = float(data[N+3])

nP = len(head)
fig = plt.figure(figsize=(16,16))
#plt.title(title)
plt.axis('off')
for i in range(nP):
    n = int(330+i+1)
    ax = fig.add_subplot(n,projection='3d')
    ax.plot_surface(X,Y,D[head[i]].T,cmap=cm.coolwarm)
    ax.set_title(ans)
    ax.set_xlabel("J2")
    ax.set_ylabel("J3")
    ax.set_title(head[i])
plt.show()
