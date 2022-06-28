import numpy as np
import functions as fs
import matplotlib.pyplot as plt
from matplotlib import cm


ans = '3x3_1'
J1, J2, J3 = (1,0.3,0)
S = 0.5
DM = False

txt_S = '05' if S == 0.5 else '03'
txt_DM = 'DM' if DM else 'no_DM'
#savenameZZ = "SSF/SSFzz_"+ans+'_'+txt_DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
#savenameXY = "SSF/SSFxy_"+ans+'_'+txt_DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
savenameZZ = "SFs/SFzz_"+ans+'_'+txt_DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
savenameXY = "SFs/SFxy_"+ans+'_'+txt_DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
SFzz = np.load(savenameZZ)
SFxy = np.load(savenameXY)
Kx,Ky = SFzz.shape

kxg = np.linspace(-8*np.pi/3,8*np.pi/3,Kx)
kyg = np.linspace(-4*np.pi/np.sqrt(3),4*np.pi/np.sqrt(3),Ky)
K = np.zeros((2,Kx,Ky))
for i in range(Kx):
    for j in range(Ky):
        K[:,i,j] = np.array([kxg[i],kyg[j]])
#
plt.figure(figsize=(16,8))
title = ans+'_'+txt_DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+')'
plt.title(title)
plt.axis('off')
plt.subplot(1,2,1)
plt.title(title+'--Sxy')
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
plt.scatter(K[0],K[1],c=SFxy,cmap = cm.plasma)
plt.colorbar()
#
plt.subplot(1,2,2)
plt.title("Szz")
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
plt.scatter(K[0],K[1],c=SFzz,cmap = cm.plasma)
plt.colorbar()
plt.show()
