import numpy as np
import matplotlib.pyplot as plt
import os
#import gaps from various files

N_ = [13,19]
ans = 'cb1'
S = 0.3
DM = False
txt_S = '05' if S == 0.5 else '03'
txt_DM = 'DM' if DM else 'no_DM'

Ji = -0.3
Jf = 0.3
g = {}
for n in N_:
    dirname = '../Data/'+str(n)+'/'+txt_S+txt_DM+'/'
    g[str(n)] = np.zeros((9,9))
    for filename in os.listdir(dirname):
        with open(dirname+filename, 'r') as f:
            lines = f.readlines()
        N = (len(lines)-1)//2 + 1
        P = []
        for i in range(N):
            data = lines[i*2+1].split(',')
            if data[0] != ans:
                continue
            if data[3] != 'True':
                print(data[3])
                print(dirname)
                g[str(n)][i2,i3] = 'nan'
                print("nan for n = ",n," and filename = ",filename)
                input()
            if data[0] == ans:
                j2 = float(data[1]) - Ji
                j3 = float(data[2]) - Ji
                i2 = int(j2*8/(0.6))
                i3 = int(j3*8/(0.6))
                g[str(n)][i2,i3] = float(data[6])
#for each point of phase diagram compute the difference between the gaps

diff = np.zeros((9,9))
for i in range(9):
    for j in range(9):
        try:
            diff[i,j] = 1 if g[str(N_[1])][i,j] - g[str(N_[0])][i,j] > 0 else 0
        except TypeError:
            print("Typerror")
            diff[i,j] = 'nan'

plt.figure()
j2 = np.linspace(Ji,Jf,9)
j3 = np.linspace(Ji,Jf,9)
X,Y = np.meshgrid(j2,j3)
plt.scatter(X,Y,c=diff)
plt.colorbar()
plt.show()
#plot points vith lower gap at higher points
