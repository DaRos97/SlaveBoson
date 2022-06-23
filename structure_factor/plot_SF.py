import numpy as np
import functions as fs
import matplotlib.pyplot as plt
from matplotlib import cm


ans = 'cb1'
J1, J2, J3 = (1,0,0)
S = 0.5
DM = False

txt_S = '05' if S == 0.5 else '03'
txt_DM = 'DM' if DM else 'no_DM'
savename = "SF_"+ans+'_'+txt_DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
SF = np.load(savename)
Kx,Ky = SF.shape
#Kx = 13     #points to compute in the SF BZ
#Ky = 13
kxg = np.linspace(-1,1,Kx)
kyg = np.linspace(-1,1,Ky)
#kxg = np.linspace(-8*np.pi/3,8*np.pi/3,Kx)
#kyg = np.linspace(-4*np.pi/np.sqrt(3),4*np.pi/np.sqrt(3),Ky)
K = np.zeros((2,Kx,Ky))
for i in range(Kx):
    for j in range(Ky):
        K[:,i,j] = np.array([kxg[i]*2*np.pi,(kxg[i]+2*kyg[j])*2*np.pi/np.sqrt(3)])
        #K = np.array([kxg[i],kyg[j]])
        #K[0,i,j] = kxg[i]
        #K[1,i,j] = kyg[i]
#
plt.figure(figsize=(8,8))
#hexagons
plt.plot(fs.X1,fs.fu1(fs.X1),'k-')
plt.hlines(2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fu3(fs.X2),'k-')
plt.plot(fs.X1,fs.fd1(fs.X1),'k-')
plt.hlines(-2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fd3(fs.X2),'k-')

plt.plot(fs.X3,fs.Fu1(fs.X3),'k-')
plt.hlines(4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fu3(fs.X4),'k-')
plt.plot(fs.X3,fs.Fd1(fs.X3),'k-')
plt.hlines(-4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fd3(fs.X4),'k-')
#
plt.scatter(K[0],K[1],c=SF,cmap = cm.plasma)#, vmin = 0, vmax = 1)
X,Y = np.meshgrid(kxg,kyg)
#plt.scatter(X,Y,c=SF,cmap = cm.plasma)#, vmin = 0, vmax = 1)
plt.colorbar()
plt.show()
