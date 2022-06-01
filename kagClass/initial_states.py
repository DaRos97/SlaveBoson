import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import time as T
from pathlib import Path
UC = 10
Nx = 17
Ny = 17
def full_func(P):
    ru = P[0]
    rx = P[1]
    rz1 = P[2]
    rz2 = P[3]
    av = P[4]
    ti = T.time()
    filename = 'FigSS/ru='+"{:.2f}".format(ru).replace('.',',')+'_rx='+"{:.2f}".format(rx).replace('.',',')+'_rz1='+"{:.2f}".format(rz1).replace('.',',')+'_rz2='+"{:.2f}".format(rz2).replace('.',',')+".png"
    if Path(filename).is_file():
        print("already computed ",filename,'\n')
        return 0
    L = np.zeros((UC,UC,3,3))       #Lx,Ly,unit cell, spin components x,y,z
    umbXtz(L,UC,ru,rx,rz1,rz2)
    K = np.ndarray((2,Nx,Ny))
    Sxy = np.ndarray((Nx,Ny))
    Sz = np.ndarray((Nx,Ny))
    S = np.ndarray((Nx,Ny))
    cntxy = 0
    cntz  = 0
    cnt = 0
    for i in range(Nx):
        for j in range(Ny):
            K[0,i,j] = -8*np.pi/3 + 16/3*np.pi/(Nx-1)*i
            K[1,i,j] = -4*np.pi/np.sqrt(3) + 8*np.pi/np.sqrt(3)/(Ny-1)*j
            Sxy[i,j], Sz[i,j] = Ss(K[:,i,j],L)
            coeff = is_inside(K[:,i,j])
#            print(K[:,i,j],coeff)
#            input()
            S[i,j] = Sxy[i,j] + Sz[i,j]
            cntxy += coeff*Sxy[i,j]
            cntz  += coeff*Sz[i,j]
            cnt   += coeff*S[i,j]
    if cntxy != 0:
        Sxy /= cntxy
    if cntz != 0:
        Sz  /= cntz
    if cnt != 0:
        S   /= cnt
    plot(K,Sxy,Sz,S,ru,rx,rz1,rz2,UC,av)
    #print("Time taken: ",T.time()-ti)
    return 0
def Ss(k,L):
    resxy = 0
    resz = 0
    for i in range(UC):
        for i2 in range(UC):
            for j in range(UC):
                for j2 in range(UC):
                    for l in range(3):
                        for l2 in range(3):
                            Li = L[i,j,l]
                            dist = np.zeros(2)
                            dist[0] = i2 - i + j2/2 - j/2 + (l2%2)/2 - (l%2)/2 + (l2//2)/4 - (l//2)/4
                            dist[1] = j2/2*np.sqrt(3) - j/2*np.sqrt(3) + (l2//2)/4*np.sqrt(3) - (l//2)/4*np.sqrt(3)
                            SiSjxy = Li[0]*L[i2,j2,l2,0] + Li[1]*L[i2,j2,l2,1]#np.dot(Li,L[i2,j2,l2])
                            SiSjz = Li[2]*L[i2,j2,l2,2]
                            resxy += np.cos(np.dot(k,dist))*SiSjxy
                            resz += np.cos(np.dot(k,dist))*SiSjz
    return resxy, resz
#z rotations
def R_z(t):
    R = np.zeros((3,3))
    R[0,0] = np.cos(t)
    R[0,1] = -np.sin(t)
    R[1,0] = np.sin(t)
    R[1,1] = np.cos(t)
    R[2,2] = 1
    return R
#z rotations
def R_x(t):
    R = np.zeros((3,3))
    R[1,1] = np.cos(t)
    R[1,2] = -np.sin(t)
    R[2,1] = np.sin(t)
    R[2,2] = np.cos(t)
    R[0,0] = 1
    return R

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
    a = np.array([-np.sqrt(3)/2,-1/2,0])
    b = np.array([np.sqrt(3)/2,-1/2,0])
    c = np.array([0,1,0])
    L[0,0,0] = a
    L[0,0,1] = b
    L[0,0,2] = c
    for i in range(0,UC):
        for j in range(0,UC):
            t = -2*np.pi/3*(i%3) + 2*np.pi/3*(j%3)
            R = R_z(t)
            L[i,j,0] = np.tensordot(R,a,1)
            L[i,j,1] = np.tensordot(R,b,1)
            L[i,j,2] = np.tensordot(R,c,1)

def octa(L):
    a = (1,0,0)
    b = (0,1,0)
    c = (-1,0,0)
    d = (0,-1,0)
    e = (0,0,-1)
    f = (0,0,1)
    for i in range(0,UC,2):
        for j in range(0,UC,2):
            L[i,j,0] = d;       L[i,j+1,0] = b;     
            L[i,j,1] = f;       L[i,j+1,1] = f;     
            L[i,j,2] = a;       L[i,j+1,2] = c;     
            L[i+1,j,0] = b;       L[i+1,j+1,0] = d; 
            L[i+1,j,1] = e;       L[i+1,j+1,1] = e; 
            L[i+1,j,2] = a;       L[i+1,j+1,2] = c; 

def cb1(L):
    a = (1,0,0)
    b = (1/2,np.sqrt(3)/2,0)
    c = (-1/2,np.sqrt(3)/2,0)
    d = (0,1/np.sqrt(3),np.sqrt(2/3))
    e = (-1/2,-1/(2*np.sqrt(3)),np.sqrt(2/3))
    f = (1/2,-1/(2*np.sqrt(3)),np.sqrt(2/3))
    a_ = (-1,0,0)
    b_ = (-1/2,-np.sqrt(3)/2,0)
    c_ = (1/2,-np.sqrt(3)/2,0)
    d_ = (0,-1/np.sqrt(3),-np.sqrt(2/3))
    e_ = (1/2,1/(2*np.sqrt(3)),-np.sqrt(2/3))
    f_ = (-1/2,1/(2*np.sqrt(3)),-np.sqrt(2/3))
    for i in range(0,UC,2):
        for j in range(0,UC,2):
            L[i,j,0] = c_;       L[i,j+1,0] = e;     
            L[i,j,1] = d;        L[i,j+1,1] = d_;     
            L[i,j,2] = f_;       L[i,j+1,2] = b;     
            L[i+1,j,0] = e_;       L[i+1,j+1,0] = c; 
            L[i+1,j,1] = a_;       L[i+1,j+1,1] = a; 
            L[i+1,j,2] = f;        L[i+1,j+1,2] = b_; 

def cb2(L):
    a = (1,0,0)
    b = (1/2,np.sqrt(3)/2,0)
    c = (-1/2,np.sqrt(3)/2,0)
    d = (0,1/np.sqrt(3),np.sqrt(2/3))
    e = (-1/2,-1/(2*np.sqrt(3)),np.sqrt(2/3))
    f = (1/2,-1/(2*np.sqrt(3)),np.sqrt(2/3))
    a_ = (-1,0,0)
    b_ = (-1/2,-np.sqrt(3)/2,0)
    c_ = (1/2,-np.sqrt(3)/2,0)
    d_ = (0,-1/np.sqrt(3),-np.sqrt(2/3))
    e_ = (1/2,1/(2*np.sqrt(3)),-np.sqrt(2/3))
    f_ = (-1/2,1/(2*np.sqrt(3)),-np.sqrt(2/3))
    for i in range(0,UC,2):
        for j in range(0,UC,2):
            L[i,j,0] = e;       L[i,j+1,0] = c_;     
            L[i,j,1] = d;       L[i,j+1,1] = d_;     
            L[i,j,2] = f;       L[i,j+1,2] = b_;     
            L[i+1,j,0] = c;        L[i+1,j+1,0] = e_; 
            L[i+1,j,1] = a_;       L[i+1,j+1,1] = a; 
            L[i+1,j,2] = f_;       L[i+1,j+1,2] = b; 

def umbq0(L,s):
    a = np.array([-np.sin(s)*np.sqrt(3)/2,-np.sin(s)/2,np.cos(s)])
    b = np.array([np.sin(s)*np.sqrt(3)/2,-np.sin(s)/2,np.cos(s)])
    c = np.array([0,np.sin(s),np.cos(s)])
    for i in range(0,UC):
        for j in range(0,UC):
            L[i,j,0] = a
            L[i,j,1] = b
            L[i,j,2] = c
def umb3x3(L,s):
    a = np.array([-np.sin(s)*np.sqrt(3)/2,-np.sin(s)/2,np.cos(s)])
    b = np.array([np.sin(s)*np.sqrt(3)/2,-np.sin(s)/2,np.cos(s)])
    c = np.array([0,np.sin(s),np.cos(s)])
    L[0,0,0] = a
    L[0,0,1] = b
    L[0,0,2] = c
    for i in range(0,UC):
        for j in range(0,UC):
            t = -2*np.pi/3*(i%3) + 2*np.pi/3*(j%3)
            R = R_z(t)
            L[i,j,0] = np.tensordot(R,a,1)
            L[i,j,1] = np.tensordot(R,b,1)
            L[i,j,2] = np.tensordot(R,c,1)

def umbX(L,s):  #umbrella along X with 3x3 order b/w unit cells
    b = np.array([np.cos(s),0,np.sin(s)])
    b = np.array([np.cos(s),np.sin(s)*np.sqrt(3)/2,-np.sin(s)/2])
    c = np.array([np.cos(s),-np.sin(s)*np.sqrt(3)/2,-np.sin(s)/2])
    L[0,0,0] = a
    L[0,0,1] = b
    L[0,0,2] = c
    for i in range(0,UC):
        for j in range(0,UC):
            t = -2*np.pi/3*(i%3) + 2*np.pi/3*(j%3)
            R = R_z(t)
            L[i,j,0] = np.tensordot(R,a,1)
            L[i,j,1] = np.tensordot(R,b,1)
            L[i,j,2] = np.tensordot(R,c,1)
######################################################
def umbXt(L,UC,ru,rx,rz):  #umbrella along X with 3x3 order b/w unit cells
    a = np.array([np.cos(ru),0,-np.sin(ru)])
    b = np.array([np.cos(ru),np.sin(ru)*np.sqrt(3)/2,np.sin(ru)/2])
    c = np.array([np.cos(ru),-np.sin(ru)*np.sqrt(3)/2,np.sin(ru)/2])
    a1 = np.tensordot(R_x(rx),a,1)
    L[0,0,0] = np.tensordot(R_x(rx),b,1)
    L[0,0,1] = np.tensordot(R_z(rz),a1,1)
    L[0,0,2] = np.tensordot(R_x(rx),c,1)
    for i in range(0,UC):
        for j in range(0,UC):
            t = -2*np.pi/3*(i%3) + 2*np.pi/3*(j%3)
            R = R_z(t)
            L[i,j,0] = np.tensordot(R,L[0,0,0],1)
            L[i,j,1] = np.tensordot(R,L[0,0,1],1)
            L[i,j,2] = np.tensordot(R,L[0,0,2],1)
######################################################
def umbXtz(L,UC,ru,rx,rz1,rz2):  #umbrella along X with 3x3 order b/w unit cells
    a = np.array([np.cos(ru),0,-np.sin(ru)])
    b = np.array([np.cos(ru),np.sin(ru)*np.sqrt(3)/2,np.sin(ru)/2])
    c = np.array([np.cos(ru),-np.sin(ru)*np.sqrt(3)/2,np.sin(ru)/2])
    a1 = np.tensordot(R_x(rx),a,1)
    b1 = np.tensordot(R_x(rx),b,1)
    c1 = np.tensordot(R_x(rx),c,1)
    L[0,0,0] = np.tensordot(R_z(rz1),b1,1)
    L[0,0,1] = a1
    L[0,0,2] = np.tensordot(R_z(rz2),c1,1)
    for i in range(0,UC):
        for j in range(0,UC):
            t = -2*np.pi/3*(i%3) + 2*np.pi/3*(j%3)
            R = R_z(t)
            L[i,j,0] = np.tensordot(R,L[0,0,0],1)
            L[i,j,1] = np.tensordot(R,L[0,0,1],1)
            L[i,j,2] = np.tensordot(R,L[0,0,2],1)
######################################################
#function that says if you are inside the EBZ or not
def is_inside(k):
    a = np.sqrt(3)
    b = 8*np.pi/np.sqrt(3)
    d = 1e-3
    kx, ky = tuple(k)
    #strictly inside or on border
    if kx < (-4*np.pi/3 - d) and kx > -8*np.pi/3 + d:
        if ky > (-a*kx-b + d) and ky < (a*kx+b - d):
            return 1
        elif np.abs(ky +a*kx+b) < d or np.abs(ky -a*kx-b) < d:
            return 1/2
    if kx > -4*np.pi/3 + d and kx < 4*np.pi/3 - d:
        if ky > -4*np.pi/np.sqrt(3) + d and ky < 4*np.pi/np.sqrt(3) - d:
            return 1
        elif np.abs(ky + 4*np.pi/np.sqrt(3)) < d or np.abs(ky - 4*np.pi/np.sqrt(3)) < d:
            return 1/2
    if kx > (4*np.pi/3 + d) and kx < 8*np.pi/3 - d:
        if ky > (a*kx-b + d) and ky < (-a*kx+b - d):
            return 1
        elif np.abs(ky -a*kx+b) < d or np.abs(ky +a*kx-b) < d:
            return 1/2
    #strictly on vertex
    if np.abs(kx + 4*np.pi/3) < d and np.abs(ky + 4*np.pi/np.sqrt(3)) < d:
        return 1/3
    if np.abs(kx - 4*np.pi/3) < d and np.abs(ky + 4*np.pi/np.sqrt(3)) < d:
        return 1/3
    if np.abs(kx + 4*np.pi/3) < d and np.abs(ky - 4*np.pi/np.sqrt(3)) < d:
        return 1/3
    if np.abs(kx - 4*np.pi/3) < d and np.abs(ky - 4*np.pi/np.sqrt(3)) < d:
        return 1/3
    if np.abs(kx + 8*np.pi/3) < d and np.abs(ky) < d:
        return 1/3
    if np.abs(kx - 8*np.pi/3) < d and np.abs(ky) < d:
        return 1/3
    return 0

######################à PLOT
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

def plot(K,Sxy,Sz,S,ru,rx,rz1,rz2,UC,av):
    dirname = 'Good/' if av else 'FigSS/'
    filename = dirname+'ru='+"{:.2f}".format(ru).replace('.',',')+'rx='+"{:.2f}".format(rx).replace('.',',')+'rz1='+"{:.2f}".format(rz1).replace('.',',')+'_rz2='+"{:.2f}".format(rz2).replace('.',',')
    title = "theta_umbrella = "+"{:.2f}".format(ru)+"\ntheta_x = "+"{:.2f}".format(rx)+"\ntheta_z1 = "+"{:.2f}".format(rz1)+"\ntheta_z2 = "+"{:.2f}".format(rz2)+"\nSites : "+str(UC**2*3)
    #figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()
    n = 3 if av else 2
    nx = 12 if av else 10
    fig = plt.figure(figsize=(nx,8))
    plt.subplot(2,n,1)
    plt.title('S_xy')

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

    plt.scatter(K[0],K[1],c=Sxy,cmap = cm.coolwarm)
    plt.colorbar()

    plt.subplot(2,n,2)
    plt.title('S_z')

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

    plt.scatter(K[0],K[1],c=Sz,cmap = cm.coolwarm)
    plt.colorbar()

    plt.subplot(2,n,3)
    plt.title('S_tot')

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

#S = Sxy + Sz
    plt.scatter(K[0],K[1],c=S, cmap=cm.coolwarm)
    plt.colorbar()
    
    plt.subplot(2,n,4)
    plt.axis('off')
    plt.text(0.2,0.5,title)
    if av:
        a = np.array([np.cos(ru),0,-np.sin(ru)])
        b = np.array([np.cos(ru),np.sin(ru)*np.sqrt(3)/2,np.sin(ru)/2])
        c = np.array([np.cos(ru),-np.sin(ru)*np.sqrt(3)/2,np.sin(ru)/2])
        #
        a1 = np.tensordot(R_x(rx),a,1)
        b1 = np.tensordot(R_x(rx),b,1)
        c1 = np.tensordot(R_x(rx),c,1)
        #
        b2 = np.tensordot(R_z(rz1),b1,1)
        a2 = a1
        c2 = np.tensordot(R_z(rz2),c1,1)
        #
        ax = fig.add_subplot(235, projection='3d')
        V = [[a2,'r'],[b2,'g'],[c2,'b']]
        for p in V:
            ax.quiver(0,0,0,p[0][0],p[0][1],p[0][2],color = p[1])
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        #
        plt.subplot(2,3,6)
        plt.axis('off')
        ch = np.dot(np.cross(a1,b1),c1)
        txt = 'Scalar chirality :'+"{:.5f}".format(ch)
        plt.text(0.2,0.5,txt)
        #plt.show()
    #
    plt.savefig(filename)
    plt.close()
    #plt.show()
