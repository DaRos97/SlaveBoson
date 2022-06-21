import numpy as np
import functions as fs
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

UC = 6
Nx = 17
Ny = 17
#q0
the = -np.pi/3
a = np.array([0,0,1/2])
b = np.array([0,-np.sqrt(3)/4,-1/4])
c = np.array([0,np.sqrt(3)/4,-1/4])
#cb1
the = int(sys.argv[1])*np.pi*2/12
phi = int(sys.argv[2])*np.pi*2/12
t0 = np.arccos(1/np.sqrt(3))
a_ = np.array([1/2,0,0])
b_ = np.array([1/4,np.sqrt(3)/4,0])
c_ = np.array([-1/4,np.sqrt(3)/4,0])
d_ = np.array([0,1/2*np.cos(t0),1/2*np.sin(t0)])
e_ = np.array([-1/2*np.cos(t0)*np.sqrt(3)/2,-1/2*np.cos(t0)/2,1/2*np.sin(t0)])
f_ = np.array([1/2*np.cos(t0)*np.sqrt(3)/2,-1/2*np.cos(t0)/2,1/2*np.sin(t0)])
#vectors
#v = [a,b,c]
#ang = [the]
v = [a_,b_,c_,d_,e_,f_]
ang = [the,phi]
L = fs.initial_state(v,ang,UC)
#
K = np.ndarray((2,Nx,Ny))
Sxy = np.ndarray((Nx,Ny))
Sz = np.ndarray((Nx,Ny))
for i in range(Nx):
    for j in range(Ny):
        K[0,i,j] = -8*np.pi/3 + 16/3*np.pi/(Nx-1)*i
        K[1,i,j] = -4*np.pi/np.sqrt(3) + 8*np.pi/np.sqrt(3)/(Ny-1)*j
        Sxy[i,j], Sz[i,j] = fs.Ss(K[:,i,j],L,UC)




fig = plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.title('S_xy')

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

plt.scatter(K[0],K[1],c=Sxy,cmap = cm.plasma)#, vmin = 0, vmax = 1)
plt.colorbar()

plt.subplot(1,2,2)
plt.title('S_z')

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

plt.scatter(K[0],K[1],c=Sz,cmap = cm.plasma)#, vmin = 0, vmax = 1)
plt.colorbar()
dirname = "Figs/"
title = dirname+"theta="+"{:.5f}".format(the).replace('.',',')+"phi="+"{:.5f}".format(phi).replace('.',',')+".png"
plt.savefig(title)
plt.show()
