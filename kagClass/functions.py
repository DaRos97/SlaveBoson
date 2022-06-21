import numpy as np
import matplotlib.pyplot as plt

#
def initial_state(v,ang,UC):
    L = np.zeros((UC,UC,3,3))
    #Ry = R_y(ang[0])
    #a = np.tensordot(Ry,v[0],1)
    #b = np.tensordot(Ry,v[1],1)
    #c = np.tensordot(Ry,v[2],1)
    #for i in range(0,UC):
    #    for j in range(0,UC):
    #        t = -2*np.pi/3*(i%3) + 2*np.pi/3*(j%3)
    #        a_ = np.tensordot(R_z(0+t),a,1)
    #        b_ = np.tensordot(R_z(-2*np.pi/3+t),b,1)
    #        c_ = np.tensordot(R_z(2*np.pi/3+t),c,1)
    #        L[i,j,0] = b_
    #        L[i,j,1] = c_
    #        L[i,j,2] = a_
    Ry = R_y(ang[0])
    Rz = R_z(ang[1])
    a = np.tensordot(Rz,np.tensordot(Ry,v[0],1),1)
    b = np.tensordot(Rz,np.tensordot(Ry,v[1],1),1)
    c = np.tensordot(Rz,np.tensordot(Ry,v[2],1),1)
    d = np.tensordot(Rz,np.tensordot(Ry,v[3],1),1)
    e = np.tensordot(Rz,np.tensordot(Ry,v[4],1),1)
    f = np.tensordot(Rz,np.tensordot(Ry,v[5],1),1)
    for i in range(0,UC,2):
        for j in range(0,UC,2):
            t1 = -2*np.pi/3*(i//2%3-j//2%3)
            B = np.tensordot(R_z(0+t1),b,1)
            E = np.tensordot(R_z(-2*np.pi/3+t1),e,1)
            D_ = np.tensordot(R_z(2*np.pi/3+t1),-d,1)
            L[i,j,0] = E
            L[i,j,1] = D_
            L[i,j,2] = B
            t2 = t1 + 2*np.pi/3
            B_ = np.tensordot(R_z(0+t2),-b,1)
            C = np.tensordot(R_z(-2*np.pi/3+t2),c,1)
            A = np.tensordot(R_z(2*np.pi/3+t2),a,1)
            L[i+1,j,0] = C
            L[i+1,j,1] = A
            L[i+1,j,2] = B_
            t3 = t1 - 2*np.pi/3
            F_ = np.tensordot(R_z(0+t3),-f,1)
            C_ = np.tensordot(R_z(-2*np.pi/3+t3),-c,1)
            D = np.tensordot(R_z(2*np.pi/3+t3),d,1)
            L[i,j+1,0] = C_
            L[i,j+1,1] = D
            L[i,j+1,2] = F_
            t4 = t1
            F = np.tensordot(R_z(0+t4),f,1)
            E_ = np.tensordot(R_z(-2*np.pi/3+t4),-e,1)
            A_ = np.tensordot(R_z(2*np.pi/3+t4),-a,1)
            L[i+1,j+1,0] = E_
            L[i+1,j+1,1] = A_
            L[i+1,j+1,2] = F

    return L
#
def Ss(k,L,UC):
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
#x rotations
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
#
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
