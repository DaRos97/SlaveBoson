import numpy as np
import inputs as inp
import matplotlib.pyplot as plt
import os
import sys
from matplotlib import cm

Color = {'3x3_1': ['b','orange'],
         'q0_1':  ['r','y'],
         'cb1':  ['g','m']}
#dirname = '../Data/S05/DM_13/'; title = "S = 0.5, with DM"
dirname = '../Data/S03/DM_13/'; title = "S = 0.366, with DM"
#
D = np.ndarray((9,9),dtype='object')
Ji = -0.3
Jf = 0.3
J2 = np.linspace(Ji,Jf,9)
J3 = np.linspace(Ji,Jf,9)
X,Y = np.meshgrid(J2,J3)
for filename in os.listdir(dirname):
    with open(dirname+filename, 'r') as f:
        lines = f.readlines()
    N = (len(lines)-1)//2 + 1
    minE = 0
    for i in range(N):
        data = lines[i*2+1].split(',')
        if data[0] not in Color.keys():
            continue
        j2 = float(data[1]) - Ji
        j3 = float(data[2]) - Ji
        i2 = int(j2*8/(0.6))
        i3 = int(j3*8/(0.6))
        if float(data[4]) < minE:
            txt_conv = 'g' if data[3] == 'True' else 'b'
            D[i2,i3] = data[0] + txt_conv
            minE = float(data[4])
##########
pts = len(os.listdir(dirname))
plt.figure(figsize=(10,5))
plt.axis('off')
plt.subplot(1,2,1)
plt.title(title)
for i in range(9):
    for j in range(9):
        c = Color[D[i,j][:-1]][0]
        m = 'o' if D[i,j][-1] == 'g' else '*'
        J2 = -0.3+i*0.6/8
        J3 = -0.3+j*0.6/8
        plt.scatter(J2,J3,color=c,marker=m)
#Legenda
plt.subplot(1,2,2)
plt.axis('off')
for a,ans in enumerate(Color.keys()):
    txt = ans + ': LRO -> '+Color[ans][0]+', SL -> '+Color[ans][1]
    plt.text(0.1,0.1+a/5,txt)


plt.show()
