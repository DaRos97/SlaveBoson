import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import time as T
from pathlib import Path
UC = 6
Nx = 17
Ny = 17
pl = 'y'
a_0 = np.array([1/2,0,0])
b_0 = np.array([1/4,np.sqrt(3)/4,0])
c_0 = np.array([-1/4,np.sqrt(3)/4,0])
d_0 = np.array([0,1/(2*np.sqrt(3)),np.sqrt(2/3)/2])
e_0 = np.array([-1/4,-1/(4*np.sqrt(3)),np.sqrt(2/3)/2])
f_0 = np.array([1/4,-1/(4*np.sqrt(3)),np.sqrt(2/3)/2])
def full_func(P):
    #ru = P[0]
    #rx = P[1]
    #rz1 = P[2]
    #rz2 = P[3]
    #av = P[4]
    av = P[-1]
    ti = T.time()
    #filename = 'FigSS/ru='+"{:.2f}".format(ru).replace('.',',')+'_rz1='+"{:.2f}".format(rz1).replace('.',',')+'_rz2='+"{:.2f}".format(rz2).replace('.',',')+".png"
    #if Path(filename).is_file():
    #    print("already computed ",filename,'\n')
    #    return 0
    L = np.zeros((UC,UC,3,3))       #Lx,Ly,unit cell, spin components x,y,z
    U,D,V,V_ = cb1(L,UC,P)#,ru,rx,rz1,rz2)
    K = np.ndarray((2,Nx,Ny))
    Sxy = np.ndarray((Nx,Ny))
    Sz = np.ndarray((Nx,Ny))
    S = np.ndarray((Nx,Ny))
    cntxy = 0
    cntz  = 0
    for i in range(Nx):
        for j in range(Ny):
            K[0,i,j] = -8*np.pi/3 + 16/3*np.pi/(Nx-1)*i
            K[1,i,j] = -4*np.pi/np.sqrt(3) + 8*np.pi/np.sqrt(3)/(Ny-1)*j
            Sxy[i,j], Sz[i,j] = Ss(K[:,i,j],L)
            coeff = is_inside(K[:,i,j])
            cntxy += coeff*Sxy[i,j]
            cntz  += coeff*Sz[i,j]
    if cntxy != 0:
        Sxy /= cntxy
    if cntz != 0:
        Sz  /= cntz
    plot2(K,Sxy,Sz,P,UC,U,D,V,V_,av)
#    plot(K,Sxy,Sz,S,ru,rx,rz1,rz2,UC,av,U,D)
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
#y rotations
def R_y(t):
    R = np.zeros((3,3))
    R[0,0] = np.cos(t)
    R[0,2] = np.sin(t)
    R[2,0] = np.sin(t)
    R[2,2] = np.cos(t)
    R[1,1] = 1
    return R

######################################################
def umbXtz(L,UC,ru,rx,rz1,rz2):  #umbrella along X with 3x3 order b/w unit cells
    a = np.array([np.cos(ru)/2,0,-np.sin(ru)/2])
    b = np.array([np.cos(ru)/2,np.sin(ru)*np.sqrt(3)/4,np.sin(ru)/4])
    c = np.array([np.cos(ru)/2,-np.sin(ru)*np.sqrt(3)/4,np.sin(ru)/4])
    a1 = np.tensordot(R_z(0),a,1)
    b1 = np.tensordot(R_z(rz1),b,1)
    c1 = np.tensordot(R_z(rz2),c,1)
    L[0,0,0] = b1
    L[0,0,1] = a1
    L[0,0,2] = c1
    for i in range(0,UC):
        for j in range(0,UC):
            t = 2*np.pi/3*(i%3) - 2*np.pi/3*(j%3)
            R = R_z(t)
            L[i,j,0] = np.tensordot(R,L[0,0,0],1)
            L[i,j,1] = np.tensordot(R,L[0,0,1],1)
            L[i,j,2] = np.tensordot(R,L[0,0,2],1)
    upCH = np.dot(np.cross(L[0,0,0],L[0,0,1]),L[0,0,2])
    dwCH = np.dot(np.cross(L[1,0,2],L[1,1,0]),L[0,1,1])
    print("Up ch: ", upCH)
    print("Down ch: ",dwCH)
    print(a1)
    print(b1)
    print(c1)
    return upCH, dwCH
######################################################
def cb1(L,UC,P):            #not really cb1
    t0 = np.arccos(1/np.sqrt(3))
    a_ = np.array([1/2,0,0])
    b_ = np.array([1/4,np.sqrt(3)/4,0])
    c_ = np.array([-1/4,np.sqrt(3)/4,0])
    d_ = np.array([0,1/2*np.cos(t0),1/2*np.sin(t0)])
    e_ = np.array([-1/2*np.cos(t0)*np.sqrt(3)/2,-1/2*np.cos(t0)/2,1/2*np.sin(t0)])
    f_ = np.array([1/2*np.cos(t0)*np.sqrt(3)/2,-1/2*np.cos(t0)/2,1/2*np.sin(t0)])
    ########################
    the = 0
    a = np.array([0,0,1/2])
    b = np.array([0,-np.sqrt(3)/4,-1/4])
    c = np.array([0,np.sqrt(3)/4,-1/4])
    Ry = R_y(the)
    a = np.tensordot(Ry,a,1)
    b = np.tensordot(Ry,b,1)
    c = np.tensordot(Ry,c,1)
    for i in range(0,UC):
        for j in range(0,UC):
            L[i,j,0] = b
            L[i,j,1] = c
            L[i,j,2] = a
    btl = []
    btr = []
    sml = []
    smu = []
    for i in range(2):
        for j in range(2):
            btl.append(np.dot(np.cross(L[1+i,0+j,0],L[0+i,1+j,1]),L[0+i,0+j,2]))
            btr.append(np.dot(np.cross(L[0+i,0+j,1],L[1+i,0+j,2]),L[0+i,1+j,0]))
            sml.append(np.dot(np.cross(L[0+i,0+j,1],L[0+i,1+j,0]),L[0+i,0+j,2]))
            smu.append(np.dot(np.cross(L[0+i,0+j,2],L[0+i,1+j,1]),L[0+i,1+j,0]))
            print(i,j)
            print("left pointing ch: ", btl[-1])
            print("right pointing ch: ",btr[-1])
            print("small 1 ch: ", sml[-1])
            print("small 2 ch: ", smu[-1])
    #
    return 0, 0, [a_,b_,c_,d_,e_,f_], [-a_,-b_,-c_,-d_,-e_,-f_]
######################################################
q = np.pi
def cb12(L,UC,P):
    t0 = 0.5#np.arccos(1/np.sqrt(3))
    a_ = np.array([1/2,0,0])
    b_ = np.array([1/4,np.sqrt(3)/4,0])
    c_ = np.array([-1/4,np.sqrt(3)/4,0])
    d_ = np.array([0,1/2*np.cos(t0),1/2*np.sin(t0)])
    e_ = np.array([-1/2*np.cos(t0)*np.sqrt(3)/2,-1/2*np.cos(t0)/2,1/2*np.sin(t0)])
    f_ = np.array([1/2*np.cos(t0)*np.sqrt(3)/2,-1/2*np.cos(t0)/2,1/2*np.sin(t0)])
    rza = 0
    rzb = 0
    rzc = 0
    tt = 0#np.pi/5
    rzd = tt
    rze = tt
    rzf = tt
    rza_ = 0
    rzb_ = 0
    rzc_ = 0
    rzd_ = tt
    rze_ = tt
    rzf_ = tt
    a1 = np.tensordot(R_z(rza),a_,1)
    b1 = np.tensordot(R_z(rzb),b_,1)
    c1 = np.tensordot(R_z(rzc),c_,1)
    d1 = np.tensordot(R_z(rzd),d_,1)
    e1 = np.tensordot(R_z(rze),e_,1)
    f1 = np.tensordot(R_z(rzf),f_,1)
    a1_ = np.tensordot(R_z(rza_),-a_,1)
    b1_ = np.tensordot(R_z(rzb_),-b_,1)
    c1_ = np.tensordot(R_z(rzc_),-c_,1)
    d1_ = np.tensordot(R_z(rzd_),-d_,1)
    e1_ = np.tensordot(R_z(rze_),-e_,1)
    f1_ = np.tensordot(R_z(rzf_),-f_,1)
    for i in range(0,UC,2):
        for j in range(0,UC,2):
            t = 0#np.pi*(i//2)# + np.pi*(j//2) #np.pi*2/3*(i/2) + np.pi*2/3*(j/2)
            tz = 0#np.pi/4*(i//2)# + np.pi*(j//2) #np.pi*2/3*(i/2) + np.pi*2/3*(j/2)
            Rz = np.dot(R_z(tz),R_y(t))
            L[i,j,0] = np.tensordot(Rz,e1,1);           L[i,j+1,0] = np.tensordot(Rz,c1_,1);
            L[i,j,1] = np.tensordot(Rz,d1_,1);          L[i,j+1,1] = np.tensordot(Rz,d1,1);
            L[i,j,2] = np.tensordot(Rz,b1,1);           L[i,j+1,2] = np.tensordot(Rz,f1_,1);
            L[i+1,j,0] = np.tensordot(Rz,c1,1);         L[i+1,j+1,0] = np.tensordot(Rz,e1_,1);
            L[i+1,j,1] = np.tensordot(Rz,a1,1);         L[i+1,j+1,1] = np.tensordot(Rz,a1_,1);
            L[i+1,j,2] = np.tensordot(Rz,b1_,1);        L[i+1,j+1,2] = np.tensordot(Rz,f1,1);
            #L[i,j,0] = np.tensordot(Rz,e1_,1);           L[i,j+1,0] = np.tensordot(Rz,c1,1);
            #L[i,j,1] = np.tensordot(Rz,d1,1);          L[i,j+1,1] = np.tensordot(Rz,d1_,1);
            #L[i,j,2] = np.tensordot(Rz,b1,1);           L[i,j+1,2] = np.tensordot(Rz,f1_,1);
            #L[i+1,j,0] = np.tensordot(Rz,c1_,1);         L[i+1,j+1,0] = np.tensordot(Rz,e1,1);
            #L[i+1,j,1] = np.tensordot(Rz,a1,1);         L[i+1,j+1,1] = np.tensordot(Rz,a1_,1);
            #L[i+1,j,2] = np.tensordot(Rz,b1_,1);        L[i+1,j+1,2] = np.tensordot(Rz,f1,1);
    btl = []
    btr = []
    sml = []
    smu = []
    for i in range(2):
        for j in range(2):
            btl.append(np.dot(np.cross(L[1+i,0+j,0],L[0+i,1+j,1]),L[0+i,0+j,2]))
            btr.append(np.dot(np.cross(L[0+i,0+j,1],L[1+i,0+j,2]),L[0+i,1+j,0]))
            sml.append(np.dot(np.cross(L[0+i,0+j,1],L[0+i,1+j,0]),L[0+i,0+j,2]))
            smu.append(np.dot(np.cross(L[0+i,0+j,2],L[0+i,1+j,1]),L[0+i,1+j,0]))
            print(i,j)
            print("left pointing ch: ", btl[-1])
            print("right pointing ch: ",btr[-1])
            print("small 1 ch: ", sml[-1])
            print("small 2 ch: ", smu[-1])
    return 0, 0, [a1,b1,c1,d1,e1,f1], [a1_,b1_,c1_,d1_,e1_,f1_]
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
######################à PLOT
######################à PLOT
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

def plot2(K,Sxy,Sz,P,UC,U,D,V,V_,av):
    rz = P[0]
    rx = P[1]
    dirname = 'Good/' if av else 'FigSS/'
    filename = dirname+'rz='+"{:.2f}".format(rz).replace('.',',')+'rx='+"{:.2f}".format(rx).replace('.',',')
    title = "theta_z = "+"{:.2f}".format(rz)+"\nSites : "+str(UC**2*3)+"\nUp triangle chirality: "+"{:.5f}".format(U)+"\nDown triangle chirality: "+"{:.5f}".format(D)
    n = 2
    nx = 14
    fig = plt.figure(figsize=(nx,14))
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

    plt.scatter(K[0],K[1],c=Sxy,cmap = cm.plasma)#, vmin = 0, vmax = 1)
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

    plt.scatter(K[0],K[1],c=Sz,cmap = cm.plasma)#, vmin = 0, vmax = 1)
    plt.colorbar()

    plt.subplot(2,n,3)
    plt.axis('off')
    plt.text(0.2,0.5,title)
    if av:
        vV = [[V[0],'r'],[V[1],'g'],[V[2],'b'],[V[3],'y'],[V[4],'m'],[V[5],'k']]
        vV_ = [[V_[0],'r'],[V_[1],'g'],[V_[2],'b'],[V_[3],'y'],[V_[4],'m'],[V_[5],'k']]
        #
        ax = fig.add_subplot(224, projection='3d')
        for p in vV:
            ax.quiver(0,0,0,p[0][0],p[0][1],p[0][2],color = p[1])
        for p in vV_:
            ax.quiver(0,0,0,p[0][0],p[0][1],p[0][2],color = p[1], alpha = 0.3)
        ax.set_xlim([-1/2, 1/2])
        ax.set_ylim([-1/2, 1/2])
        ax.set_zlim([-1/2, 1/2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if pl == 'y':
            plt.show()
            exit()
    #
    plt.savefig(filename)
    plt.close()
    #plt.show()
###################################################################################
###################################################################################
###################################################################################
###################################################################################
def plot(K,Sxy,Sz,S,ru,rx,rz1,rz2,UC,av,U,D):
    dirname = 'Good/' if av else 'FigSS/'
    filename = dirname+'ru='+"{:.2f}".format(ru).replace('.',',')+'rz1='+"{:.2f}".format(rz1).replace('.',',')+'_rz2='+"{:.2f}".format(rz2).replace('.',',')
    title = "theta_umbrella = "+"{:.2f}".format(ru)+"\ntheta_z1 = "+"{:.2f}".format(rz1)+"\ntheta_z2 = "+"{:.2f}".format(rz2)+"\nSites : "+str(UC**2*3)+"\nUp triangle chirality: "+"{:.5f}".format(U)+"\nDown triangle chirality: "+"{:.5f}".format(D)
    #figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()
    n = 2 if av else 2
    nx = 10 if av else 10
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

    plt.scatter(K[0],K[1],c=Sxy,cmap = cm.plasma)#, vmin = 0, vmax = 1)
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

    plt.scatter(K[0],K[1],c=Sz,cmap = cm.plasma)#, vmin = 0, vmax = 1)
    plt.colorbar()

    plt.subplot(2,n,3)
    plt.axis('off')
    plt.text(0.2,0.5,title)
    if av:
        a = np.array([np.cos(ru)/2,0,-np.sin(ru)/2])
        b = np.array([np.cos(ru)/2,np.sin(ru)*np.sqrt(3)/4,np.sin(ru)/4])
        c = np.array([np.cos(ru)/2,-np.sin(ru)*np.sqrt(3)/4,np.sin(ru)/4])
        #
        a1 = np.tensordot(R_x(rx),a,1)
        b1 = np.tensordot(R_x(rx),b,1)
        c1 = np.tensordot(R_x(rx),c,1)
        #
        b2 = np.tensordot(R_z(rz1),b1,1)
        a2 = a1
        c2 = np.tensordot(R_z(rz2),c1,1)
        #
        ax = fig.add_subplot(224, projection='3d')
        V = [[a2,'r'],[b2,'g'],[c2,'b']]
        for p in V:
            ax.quiver(0,0,0,p[0][0],p[0][1],p[0][2],color = p[1])
        ax.set_xlim([-1/2, 1/2])
        ax.set_ylim([-1/2, 1/2])
        ax.set_zlim([-1/2, 1/2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        _A = pl#input("Show? [y/n]")
        if _A == 'y':
            plt.show()
            exit()
    #
    plt.savefig(filename)
    plt.close()
    #plt.show()

######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
a = np.array([1/2,0,0])
b = np.array([1/4,np.sqrt(3)/4,0])
c = np.array([-1/4,np.sqrt(3)/4,0])
d = np.array([0,1/(2*np.sqrt(3)),np.sqrt(2/3)/2])
e = np.array([-1/4,-1/(4*np.sqrt(3)),np.sqrt(2/3)/2])
f = np.array([1/4,-1/(4*np.sqrt(3)),np.sqrt(2/3)/2])
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################






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
