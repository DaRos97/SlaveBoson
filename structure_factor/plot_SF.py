import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


ans = '3x3_1'
J1, J2, J3 = (1,0.225,0)
S = 0.5
DM = False

txt_S = '05' if S == 0.5 else '03'
txt_DM = 'DM' if DM else 'no_DM'
savename = "SF_"+ans+'_'+txt_DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
SF = np.load(savename)


Kx = 13     #points to compute in the SF BZ
Ky = 13
kxg = np.linspace(0,1,Kx)
kyg = np.linspace(0,1,Ky)
K = np.zeros((2,Kx,Ky))
for i in range(Kx):
    for j in range(Ky):
        K[:,i,j] = np.array([kxg[i]*2*np.pi,(kxg[i]+kyg[j])*2*np.pi/np.sqrt(3)])

plt.figure(figsize=(8,8))
plt.scatter(K[0],K[1],c=SF,cmap = cm.plasma)#, vmin = 0, vmax = 1)
plt.colorbar()
plt.show()
