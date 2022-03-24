import inputs as inp
import numpy as np
from scipy import linalg as LA

grid_pts = inp.grid_pts
#### vectors of 1nn, 2nn and 3nn
e1 = (1/4,np.sqrt(3)/4);    e1_ = (-1/4,-np.sqrt(3)/4)
e2 = (1/4,-np.sqrt(3)/4);   e2_ = (-1/4,np.sqrt(3)/4)
e3 = (-1/2,0);              e3_ = (1/2,0)
f1 = (3/4,-np.sqrt(3)/4);   f1_ = (-3/4,np.sqrt(3)/4)
f2 = (-3/4,-np.sqrt(3)/4);  f2_ = (3/4,np.sqrt(3)/4)
f3 = (0,np.sqrt(3)/2);      f3_ = (0,-np.sqrt(3)/2)
g1 = (-1/2,-np.sqrt(3)/2);  g1_ = (1/2,np.sqrt(3)/2)
g2 = (-1/2,np.sqrt(3)/2);   g2_ = (1/2,-np.sqrt(3)/2)
g3 = (1,0);                 g3_ = (-1,0)
####
def exp_k(a):
    ax, ay = a
    res = np.ndarray((grid_pts,grid_pts),dtype=complex)
    for i in range(grid_pts):
        for j in range(grid_pts):
            res[i,j] = np.exp(1j*(inp.kg[0][i]*ax+inp.kg[1][j]*ay))
    return res
#### all ansatze
def Nk(P,L,args):
    m = 6
    J1,J2,J3,ans = args
    J1 /= 2.
    J2 /= 2.
    J3 /= 2.
    #params
    if ans == '3x3':
        A1,A3,B1,B2,B3 = P
        A2 = 0
        phiA1p, phiA2, phiA2p, phiA3 = (np.pi, 0, 0, 0)
        phiB1, phiB1p, phiB2, phiB2p, phiB3 = (np.pi, np.pi, 0, 0, np.pi)
        p1 = 0
    elif ans == 'q0':
        A1,A2,B1,B2,B3 = P
        A3 = 0
        phiA1p, phiA2, phiA2p, phiA3 = (0, 0, 0, 0)
        phiB1, phiB1p, phiB2, phiB2p, phiB3 = (np.pi, np.pi, np.pi, np.pi, 0)
        p1 = 0
    elif ans == '0-pi':
        A1,A2,A3,B1,B2 = P
        B3 = 0
        phiA1p, phiA2, phiA2p, phiA3 = (0, 0, 0, np.pi)
        phiB1, phiB1p, phiB2, phiB2p, phiB3 = (np.pi, np.pi, np.pi, np.pi, 0)
        p1 = 1
    elif ans == 'pi-pi':
        A1,B1,B2 = P
        A2,A3,B3 = (0, 0, 0)
        phiA1p, phiA2, phiA2p, phiA3 = (np.pi, 0, np.pi, 0)
        phiB1, phiB1p, phiB2, phiB2p, phiB3 = (np.pi, np.pi, 0, 0, 0)
        p1 = 1
    elif ans == 'cb1':
        A1,A2,A3,B1,B2,B3,phiA1p,phiB2 = P
        phiA2, phiA2p, phiA3 = (phiA1p/2, phiA1p/2, (phiA1p+np.pi)/2)
        phiB1, phiB1p, phiB2p, phiB3 = (np.pi, np.pi, -phiB2, 0)
        p1 = 1
    ################
    N = np.zeros((2*m,2*m,grid_pts,grid_pts), dtype=complex)
    ##################################### B and L
    b1 = B1*np.exp(1j*phiB1);               b1_ = np.conjugate(b1)
    b1p = B1*np.exp(1j*phiB1p);             b1p_ = np.conjugate(b1p)
    b1pi = B1*np.exp(1j*(phiB1p+p1*np.pi)); b1pi_ = np.conjugate(b1pi)
    b2 = B2*np.exp(1j*phiB2);               b2_ = np.conjugate(b2)
    b2i = B2*np.exp(1j*(phiB2+p1*np.pi));   b2i_ = np.conjugate(b2i)
    b2p = B2*np.exp(1j*phiB2p);             b2p_ = np.conjugate(b2p)
    b2pi = B2*np.exp(1j*(phiB2p+p1*np.pi)); b2pi_ = np.conjugate(b2pi)
    N[0,1] = J1*b1p_*exp_k(e1)  + J2*b2_*exp_k(f1_)
    N[0,2] = J1*b1p*exp_k(e2_)  + J2*b2p*exp_k(f2_)
    N[0,4] = J1*b1_*exp_k(e1_)  + J2*exp_k(f1)*(b2p_ if not p1 else b2p)
    N[0,5] = J1*b1*exp_k(e2)    + J2*exp_k(f2)*(b2 if not p1 else b2_)
    N[1,2] = J1*(b1_*exp_k(e3_) + b1p_*exp_k(e3))
    N[1,3] = J1*b1*exp_k(e1)    + J2*exp_k(f1_)*(b2pi if not p1 else b2pi_)
    N[1,5] = J2*(b2_*exp_k(f3_) + b2p_*exp_k(f3))
    N[2,3] = J1*b1_*exp_k(e2_)  + J2*exp_k(f2_)*(b2i_ if not p1 else b2i)
    N[2,4] = J2*(b2p*exp_k(f3_) + b2i*exp_k(f3))
    N[3,4] = J1*b1pi_*exp_k(e1) + J2*b2_*exp_k(f1_)
    N[3,5] = J1*b1p*exp_k(e2_)  + J2*b2pi*exp_k(f2_)
    N[4,5] = J1*(b1_*exp_k(e3_) + b1pi_*exp_k(e3))
    # B3
    b3 = B3*np.exp(1j*phiB3);               b3_ = np.conjugate(b3)
    b3i = B3*np.exp(1j*(phiB3+p1*np.pi));   b3i_ = np.conjugate(b3i)
    N[0,0] = J3*(b3_*exp_k(g3_)  + b3*exp_k(g3))
    N[3,3] = J3*(b3i_*exp_k(g3_) + b3i*exp_k(g3))
    N[1,4] = J3*(b3_*exp_k(g2_)  + b3*exp_k(g2))
    N[2,5] = J3*(b3*exp_k(g1)    + b3i_*exp_k(g1_))
    ####other half square
    N[m+0,m+1] = J1*b1p*exp_k(e1)  + J2*b2*exp_k(f1_)
    N[m+0,m+2] = J1*b1p_*exp_k(e2_)  + J2*b2p*exp_k(f2_)
    N[m+0,m+4] = J1*b1*exp_k(e1_)  + J2*exp_k(f1)*(b2p if not p1 else b2p_)
    N[m+0,m+5] = J1*b1_*exp_k(e2)    + J2*exp_k(f2)*(b2_ if not p1 else b2)
    N[m+1,m+2] = J1*(b1*exp_k(e3_) + b1p*exp_k(e3))
    N[m+1,m+3] = J1*b1_*exp_k(e1)    + J2*exp_k(f1_)*(b2pi_ if not p1 else b2pi)
    N[m+1,m+5] = J2*(b2*exp_k(f3_) + b2p*exp_k(f3))
    N[m+2,m+3] = J1*b1*exp_k(e2_)  + J2*exp_k(f2_)*(b2i if not p1 else b2i_)
    N[m+2,m+4] = J2*(b2p_*exp_k(f3_) + b2i_*exp_k(f3))
    N[m+3,m+4] = J1*b1pi*exp_k(e1) + J2*b2*exp_k(f1_)
    N[m+3,m+5] = J1*b1p_*exp_k(e2_)  + J2*b2pi_*exp_k(f2_)
    N[m+4,m+5] = J1*(b1*exp_k(e3_) + b1pi*exp_k(e3))
    N[m+0,m+0] = J3*(b3*exp_k(g3_)  + b3_*exp_k(g3))
    N[m+3,m+3] = J3*(b3i*exp_k(g3_) + b3i_*exp_k(g3))
    N[m+1,m+4] = J3*(b3*exp_k(g2_)  + b3_*exp_k(g2))
    N[m+2,m+5] = J3*(b3_*exp_k(g1)    + b3i*exp_k(g1_))
    ######################################## A
    a1 = A1
    a1p = A1*np.exp(1j*phiA1p)
    a1pi = A1*np.exp(1j*(phiA1p+p1*np.pi))
    a2 = A2*np.exp(1j*phiA2)
    a2i = A2*np.exp(1j*(phiA2+p1*np.pi))
    a2p = A2*np.exp(1j*phiA2p)
    a2pi = A2*np.exp(1j*(phiA2p+p1*np.pi))
    N[0,m+1] = - J1*a1p*exp_k(e1)   -   J2*a2*exp_k(f1_)
    N[0,m+2] =   J1*a1p*exp_k(e2_)  +   J2*a2p*exp_k(f2_)
    N[0,m+4] = - J1*a1*exp_k(e1_)   -(-1)**p1   *J2*a2p*exp_k(f1)
    N[0,m+5] =   J1*a1*exp_k(e2)    +(-1)**p1   *J2*a2*exp_k(f2)
    N[1,m+2] = -J1*(a1*exp_k(e3_) + a1p*exp_k(e3))
    N[1,m+3] =   J1*a1*exp_k(e1)    +(-1)**p1   *J2*a2pi*exp_k(f1_)
    N[1,m+5] = -J2*(a2*exp_k(f3_) + a2p*exp_k(f3))
    N[2,m+3] = - J1*a1*exp_k(e2_)   -(-1)**p1   *J2*a2i*exp_k(f2_)
    N[2,m+4] = J2*(a2p*exp_k(f3_) + a2i*exp_k(f3))
    N[3,m+4] = - J1*a1pi*exp_k(e1)   -   J2*a2*exp_k(f1_)
    N[3,m+5] =   J1*a1p*exp_k(e2_)  +   J2*a2pi*exp_k(f2_)
    N[4,m+5] = -J1*(a1*exp_k(e3_) + a1pi*exp_k(e3))
    # A3
    a3 = A3*np.exp(1j*phiA3)
    a3i = A3*np.exp(1j*(phiA3+p1*np.pi))
    N[0,m+0] = J3*(a3*exp_k(g3) - a3*exp_k(g3_))
    N[1,m+4] = J3*(a3*exp_k(g2) - a3*exp_k(g2_))
    N[2,m+5] = J3*(a3*exp_k(g1) - a3i*exp_k(g1_))
    N[3,m+3] = J3*(a3i*exp_k(g3) - a3i*exp_k(g3_))
    #not the diagonal
    for i in range(m-1):
        for j in range(m+1+i,2*m):
            N[j-m,i+m] = -np.conjugate(N[i,j])
    # HERMITIAN MATRIX
    for i in range(2*m-1):
        for j in range(1,2*m):
            N[j,i] = np.conjugate(N[i,j])
    # L
    for i in range(2*m):
        N[i,i] += L
    #multiply by tau 3
    for i in range(m,2*m):
        for j in range(2*m):
            N[i,j] *= -1
    return N


