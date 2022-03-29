import inputs as inp
import numpy as np
from scipy import linalg as LA

grid_pts = inp.grid_pts
#### vectors of 1nn, 2nn and 3nn
e1 = (1/4,np.sqrt(3)/4);    ke1 = np.exp(1j*np.tensordot(e1,inp.Mkg,axes=1));   ke1_ = np.conjugate(ke1);
e2 = (1/4,-np.sqrt(3)/4);   ke2 = np.exp(1j*np.tensordot(e2,inp.Mkg,axes=1));   ke2_ = np.conjugate(ke2);
e3 = (-1/2,0);              ke3 = np.exp(1j*np.tensordot(e3,inp.Mkg,axes=1));   ke3_ = np.conjugate(ke3);
f1 = (3/4,-np.sqrt(3)/4);   kf1 = np.exp(1j*np.tensordot(f1,inp.Mkg,axes=1));   kf1_ = np.conjugate(kf1);
f2 = (-3/4,-np.sqrt(3)/4);  kf2 = np.exp(1j*np.tensordot(f2,inp.Mkg,axes=1));   kf2_ = np.conjugate(kf2);
f3 = (0,np.sqrt(3)/2);      kf3 = np.exp(1j*np.tensordot(f3,inp.Mkg,axes=1));   kf3_ = np.conjugate(kf3);
g1 = (-1/2,-np.sqrt(3)/2);  kg1 = np.exp(1j*np.tensordot(g1,inp.Mkg,axes=1));   kg1_ = np.conjugate(kg1);
g2 = (-1/2,np.sqrt(3)/2);   kg2 = np.exp(1j*np.tensordot(g2,inp.Mkg,axes=1));   kg2_ = np.conjugate(kg2);
g3 = (1,0);                 kg3 = np.exp(1j*np.tensordot(g3,inp.Mkg,axes=1));   kg3_ = np.conjugate(kg3);
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
        phiA1p, phiA2, phiA2p, phiA3 = (0, np.pi, np.pi, 0)
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
        A1,B1,B2,B3,phiA1p = P
        A2 = 0
        A3 = 0
        phiA2, phiA2p, phiA3 = (0,0,0)
        phiB1, phiB1p, phiB2, phiB2p, phiB3 = (np.pi, np.pi, 0, 0, 0)
        p1 = 1
    ################
    N = np.zeros((2*m,2*m,grid_pts,grid_pts), dtype=complex)
    ##################################### B
    b1 = B1*np.exp(1j*phiB1);               b1_ = np.conjugate(b1)
    b1p = B1*np.exp(1j*phiB1p);             b1p_ = np.conjugate(b1p)
    b1pi = B1*np.exp(1j*(phiB1p+p1*np.pi)); b1pi_ = np.conjugate(b1pi)
    b2 = B2*np.exp(1j*phiB2);               b2_ = np.conjugate(b2)
    b2i = B2*np.exp(1j*(phiB2+p1*np.pi));   b2i_ = np.conjugate(b2i)
    b2p = B2*np.exp(1j*phiB2p);             b2p_ = np.conjugate(b2p)
    b2pi = B2*np.exp(1j*(phiB2p+p1*np.pi)); b2pi_ = np.conjugate(b2pi)
    b3 = B3*np.exp(1j*phiB3);               b3_ = np.conjugate(b3)
    b3i = B3*np.exp(1j*(phiB3+p1*np.pi));   b3i_ = np.conjugate(b3i)
    N[0,1] = J1*b1p_*ke1  + J2*b2*kf1_
    N[0,2] = J1*b1p*ke2_  + J2*b2p_*kf2_
    N[0,4] = J1*b1_*ke1_  + J2*kf1*(b2p if not p1 else b2p_)
    N[0,5] = J1*b1*ke2    + J2*kf2*(b2_ if not p1 else b2)
    N[1,2] = J1*(b1_*ke3_ + b1p_*ke3)
    N[1,3] = J1*b1*ke1    + J2*kf1_*(b2pi_ if not p1 else b2pi)
    N[1,5] = J2*(b2*kf3_ + b2p*kf3)
    N[2,3] = J1*b1_*ke2_  + J2*kf2_*(b2i if not p1 else b2i_)
    N[2,4] = J2*(b2p_*kf3_ + b2i_*kf3)
    N[3,4] = J1*b1pi_*ke1 + J2*b2*kf1_
    N[3,5] = J1*b1p*ke2_  + J2*b2pi_*kf2_
    N[4,5] = J1*(b1_*ke3_ + b1pi_*ke3)
    N[0,0] = J3*kg3_*(b3_ if not p1 else b3)
    N[3,3] = J3*kg3_*(b3i_ if not p1 else b3i)
    N[1,4] = J3*(b3_*kg2_  + b3*kg2)
    N[2,5] = J3*(b3*kg1    + b3i_*kg1_)
    ####other half square
    N[m+0,m+1] = J1*b1p*ke1  + J2*b2_*kf1_
    N[m+0,m+2] = J1*b1p_*ke2_  + J2*b2p*kf2_
    N[m+0,m+4] = J1*b1*ke1_  + J2*kf1*(b2p_ if not p1 else b2p)
    N[m+0,m+5] = J1*b1_*ke2    + J2*kf2*(b2 if not p1 else b2_)
    N[m+1,m+2] = J1*(b1*ke3_ + b1p*ke3)
    N[m+1,m+3] = J1*b1_*ke1    + J2*kf1_*(b2pi if not p1 else b2pi_)
    N[m+1,m+5] = J2*(b2_*kf3_ + b2p_*kf3)
    N[m+2,m+3] = J1*b1*ke2_  + J2*kf2_*(b2i_ if not p1 else b2i)
    N[m+2,m+4] = J2*(b2p*kf3_ + b2i*kf3)
    N[m+3,m+4] = J1*b1pi*ke1 + J2*b2_*kf1_
    N[m+3,m+5] = J1*b1p_*ke2_  + J2*b2pi*kf2_
    N[m+4,m+5] = J1*(b1*ke3_ + b1pi*ke3)
    N[m+0,m+0] = J3*kg3_*(b3 if not p1 else b3_)
    N[m+3,m+3] = J3*kg3_*(b3i if not p1 else b3i_)
    N[m+1,m+4] = J3*(b3*kg2_  + b3_*kg2)
    N[m+2,m+5] = J3*(b3_*kg1    + b3i*kg1_)
    ######################################## A
    a1 = A1
    a1p = A1*np.exp(1j*phiA1p)
    a1pi = A1*np.exp(1j*(phiA1p+p1*np.pi))
    a2 = A2*np.exp(1j*phiA2)
    a2i = A2*np.exp(1j*(phiA2+p1*np.pi))
    a2p = A2*np.exp(1j*phiA2p)
    a2pi = A2*np.exp(1j*(phiA2p+p1*np.pi))
    a3 = A3*np.exp(1j*phiA3)
    a3i = A3*np.exp(1j*(phiA3+p1*np.pi))
    N[0,m+1] = - J1*a1p*ke1   +J2*a2*kf1_
    N[0,m+2] =   J1*a1p*ke2_  -J2*a2p*kf2_
    N[0,m+4] = - J1*a1*ke1_   +(-1)**p1   *J2*a2p*kf1
    N[0,m+5] =   J1*a1*ke2    -(-1)**p1   *J2*a2*kf2
    N[1,m+2] = - J1*(a1*ke3_  +a1p*ke3)
    N[1,m+3] =   J1*a1*ke1    -(-1)**p1   *J2*a2pi*kf1_
    N[1,m+5] =   J2*(a2*kf3_  +a2p*kf3)
    N[2,m+3] = - J1*a1*ke2_   +(-1)**p1   *J2*a2i*kf2_
    N[2,m+4] = - J2*(a2p*kf3_ +a2i*kf3)
    N[3,m+4] = - J1*a1pi*ke1  +J2*a2*kf1_
    N[3,m+5] =   J1*a1p*ke2_  -J2*a2pi*kf2_
    N[4,m+5] = - J1*(a1*ke3_  +a1pi*ke3)
    N[1,m+4] = - J3*(a3*kg2_  -a3*kg2)
    N[2,m+5] =   J3*(a3*kg1   -a3i*kg1_)
    N[0,m+0] = - J3*kg3_*a3*(-1)**p1
    N[3,m+3] = - J3*kg3_*a3i*(-1)**p1
    #not the diagonal
    N[1,m] =   J1*a1p*ke1_   -   J2*a2*kf1
    N[2,m] = -  J1*a1p*ke2  +   J2*a2p*kf2
    N[4,m] =  J1*a1*ke1   -(-1)**p1   *J2*a2p*kf1_
    N[5,m] =  - J1*a1*ke2_    +(-1)**p1   *J2*a2*kf2_
    N[2,m+1] = J1*(a1*ke3 + a1p*ke3_)
    N[3,m+1] = -  J1*a1*ke1_    +(-1)**p1   *J2*a2pi*kf1
    N[5,m+1] = -J2*(a2*kf3 + a2p*kf3_)
    N[3,m+2] =  J1*a1*ke2   -(-1)**p1   *J2*a2i*kf2
    N[4,m+2] = J2*(a2p*kf3 + a2i*kf3_)
    N[4,m+3] =  J1*a1pi*ke1_   -   J2*a2*kf1
    N[5,m+3] =  - J1*a1p*ke2  +   J2*a2pi*kf2
    N[5,m+4] = J1*(a1*ke3 + a1pi*ke3_)
    N[4,m+1] = J3*(a3*kg2 - a3*kg2_)
    N[5,m+2] = -J3*(a3*kg1_ - a3i*kg1)
    N[0,m+0] += J3*kg3*a3*(-1)**p1
    N[3,m+3] += J3*kg3*a3i*(-1)**p1
    #################################### HERMITIAN MATRIX
    for i in range(2*m):
        for j in range(i,2*m):
            N[j,i] += np.conjugate(N[i,j])
    #################################### L
    for i in range(2*m):
        N[i,i] += L
    ####################################multiply by tau 3
    for i in range(m,2*m):
        for j in range(2*m):
            N[i,j] *= -1
    return N


