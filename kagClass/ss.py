import numpy as np
import matplotlib.pyplot as plt


UC = 9
L = np.zeros((UC,UC,3,3))       #Lx,Ly,unit cell, spin components x,y,z
#fill lattice
def q0(L):
    for i in range(UC):
        for j in range(UC):
            L[i,j,0] = (-np.sqrt(3)/2,-1/2,0)
            L[i,j,1] = (np.sqrt(3)/2,-1/2,0)
            L[i,j,2] = (0,1,0)
def ferro(L):
    for i in range(UC):
        for j in range(UC):
            for l in range(3):
                L[i,j,l,2] = 1
def s3x3(L):
    a = (-np.sqrt(3)/2,-1/2,0)
    b = (np.sqrt(3)/2,-1/2,0)
    c = (0,1,0)
    for i in range(0,UC,3):
        for j in range(0,UC,3):
            L[i,j,0] = c;       L[i,j+1,0] = a;     L[i,j+2,0] = b
            L[i,j,1] = a;       L[i,j+1,1] = b;     L[i,j+2,1] = c
            L[i,j,2] = b;       L[i,j+1,2] = c;     L[i,j+2,2] = a
            L[i+1,j,0] = b;       L[i+1,j+1,0] = c;     L[i+1,j+2,0] = a
            L[i+1,j,1] = c;       L[i+1,j+1,1] = a;     L[i+1,j+2,1] = b
            L[i+1,j,2] = a;       L[i+1,j+1,2] = b;     L[i+1,j+2,2] = c
            L[i+2,j,0] = a;       L[i+2,j+1,0] = b;     L[i+2,j+2,0] = c
            L[i+2,j,1] = b;       L[i+2,j+1,1] = c;     L[i+2,j+2,1] = a
            L[i+2,j,2] = c;       L[i+2,j+1,2] = a;     L[i+2,j+2,2] = b

def octa(L):
    a = (-np.sqrt(3)/2,-1/2,0)
    b = (np.sqrt(3)/2,-1/2,0)
    c = (0,1,0)

    for i in range(0,UC,3):
        for j in range(0,UC,3):
            L[i,j,0] = c;       L[i,j+1,0] = a;     L[i,j+2,0] = b
            L[i,j,1] = a;       L[i,j+1,1] = b;     L[i,j+2,1] = c
            L[i,j,2] = b;       L[i,j+1,2] = c;     L[i,j+2,2] = a
            L[i+1,j,0] = b;       L[i+1,j+1,0] = c;     L[i+1,j+2,0] = a
            L[i+1,j,1] = c;       L[i+1,j+1,1] = a;     L[i+1,j+2,1] = b
            L[i+1,j,2] = a;       L[i+1,j+1,2] = b;     L[i+1,j+2,2] = c
            L[i+2,j,0] = a;       L[i+2,j+1,0] = b;     L[i+2,j+2,0] = c
            L[i+2,j,1] = b;       L[i+2,j+1,1] = c;     L[i+2,j+2,1] = a
            L[i+2,j,2] = c;       L[i+2,j+1,2] = a;     L[i+2,j+2,2] = b


s3x3(L)

def Ss(k):
    resxy = 0
    resz = 0
    for i in range(UC):
        for i2 in range(i,UC):
            for j in range(UC):
                for j2 in range(j,UC):
                    for l in range(3):
                        for l2 in range(l,3):
                            Li = L[i,j,l]
                            dist = np.zeros(2)
                            dist[0] = i2 - i + j2/2 - j/2 + (l2%2)/2 - (l%2)/2 + (l2//2)/4 - (l//2)/4
                            dist[1] = j2/2*np.sqrt(3) - j/2*np.sqrt(3) + (l2//2)/4*np.sqrt(3) - (l//2)/4*np.sqrt(3)
                            SiSjxy = Li[0]*L[i2,j2,l2,0] + Li[1]*L[i2,j2,l2,1]#np.dot(Li,L[i2,j2,l2])
                            SiSjz = Li[2]*L[i2,j2,l2,2]
                            resxy += np.cos(np.dot(k,dist))*SiSjxy
                            resz += np.cos(np.dot(k,dist))*SiSjz
    return resxy, resz


Nx = 17
Ny = 17

kxg = np.linspace(0,1,Nx)
kyg = np.linspace(0,1,Ny)
K = np.ndarray((2,Nx,Ny))
Sxy = np.ndarray((Nx,Ny))
Sz = np.ndarray((Nx,Ny))
for i in range(Nx):
    for j in range(Ny):
#        K[0,i,j] = kxg[i]*2*np.pi
#        K[1,i,j] = (kxg[i]+2*kyg[j])*2*np.pi/np.sqrt(3)
        K[0,i,j] = -8*np.pi/3 + 16/3*np.pi/(Nx-1)*i
        K[1,i,j] = -4*np.pi/np.sqrt(3) + 8*np.pi/np.sqrt(3)/(Ny-1)*j
#        plt.scatter(K[0,i,j],K[1,i,j],color='k')
        Sxy[i,j], Sz[i,j] = Ss(K[:,i,j])

def fd1(x):
    return -np.sqrt(3)*x-4*np.pi/np.sqrt(3)
def fd3(x):
    return np.sqrt(3)*x-4*np.pi/np.sqrt(3)
def fu1(x):
    return np.sqrt(3)*x+4*np.pi/np.sqrt(3)
def fu3(x):
    return -np.sqrt(3)*x+4*np.pi/np.sqrt(3)
def Fd1(x):
    return -np.sqrt(3)*x-8*np.pi/np.sqrt(3)
def Fd3(x):
    return np.sqrt(3)*x-8*np.pi/np.sqrt(3)
def Fu1(x):
    return np.sqrt(3)*x+8*np.pi/np.sqrt(3)
def Fu3(x):
    return -np.sqrt(3)*x+8*np.pi/np.sqrt(3)

X1 = np.linspace(-4*np.pi/3,-2*np.pi/3,1000)
X2 = np.linspace(2*np.pi/3,4*np.pi/3,1000)
X3 = np.linspace(-8*np.pi/3,-4*np.pi/3,1000)
X4 = np.linspace(4*np.pi/3,8*np.pi/3,1000)

fig = plt.figure(figsize=(16,8))
plt.subplot(1,2,1)

plt.plot(X1,fu1(X1),'k-')
plt.hlines(2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(X2,fu3(X2),'k-')
plt.plot(X1,fd1(X1),'k-')
plt.hlines(-2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(X2,fd3(X2),'k-')

plt.plot(X3,Fu1(X3),'k-')
plt.hlines(4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(X4,Fu3(X4),'k-')
plt.plot(X3,Fd1(X3),'k-')
plt.hlines(-4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(X4,Fd3(X4),'k-')

plt.scatter(K[0],K[1],c=Sxy)
plt.colorbar()

plt.subplot(1,2,2)

plt.plot(X1,fu1(X1),'k-')
plt.hlines(2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(X2,fu3(X2),'k-')
plt.plot(X1,fd1(X1),'k-')
plt.hlines(-2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(X2,fd3(X2),'k-')

plt.plot(X3,Fu1(X3),'k-')
plt.hlines(4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(X4,Fu3(X4),'k-')
plt.plot(X3,Fd1(X3),'k-')
plt.hlines(-4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(X4,Fd3(X4),'k-')

plt.scatter(K[0],K[1],c=Sz)
plt.colorbar()
plt.show()

