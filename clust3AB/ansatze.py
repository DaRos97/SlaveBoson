import inputs as inp
import numpy as np
from scipy import linalg as LA

m = inp.m
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
#### DM
t1 = np.exp(1j*inp.DM1);    t1_ = np.conjugate(t1)
t3 = np.exp(1j*inp.DM3);    t3_ = np.conjugate(t3)
#### all ansatze
def Nk(P,L,args):
    J1,J2,J3,ans = args
    J1 /= 2.
    J2 /= 2.
    J3 /= 2.
    j2 = np.sign(int(np.abs(J2)*1e8))
    j3 = np.sign(int(np.abs(J3)*1e8))
    A1 = P[0]
    #params
    if ans == '3x3':
        A2 = 0
        A3 = j3*P[1]
        B1 = P[2*j3]*j3 + P[1]*(1-j3)
        B2 = P[3*j2*j3]*j2*j3+P[2*j2*(1-j3)]*(1-j3)*j2
        B3 = P[4*j3*j2]*j3*j2+P[3*j3*(1-j2)]*j3*(1-j2)
        phiA1p, phiA2, phiA2p, phiA3 = (np.pi, 0, 0, 0)
        phiB1, phiB1p, phiB2, phiB2p, phiB3 = (np.pi, np.pi, 0, 0, np.pi)
        p1 = 0
    elif ans == 'q0':
        A2 = P[1]*j2
        A3 = 0.
        B1 = P[2*j2]*j2+P[1]*(1-j2)
        B2 = P[3*j2]*j2
        B3 = P[4*j3*j2]*j3*j2+P[2*j3*(1-j2)]*j3*(1-j2)
        phiA1p, phiA2, phiA2p, phiA3 = (0, np.pi, np.pi, 0)
        phiB1, phiB1p, phiB2, phiB2p, phiB3 = (np.pi, np.pi, np.pi, np.pi, 0)
        p1 = 0
    elif ans == '0-pi':      #A1,A2,A3,B1,B2,phiA1p,phiB2
        A2 = P[1*j2]*j2
        A3 = P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2)
        B1 = P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3)
        B2 = P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3)
        B3 = 0
        phiA1p, phiA2, phiA2p, phiA3 = (0, np.pi, np.pi, np.pi)
        phiB1, phiB1p, phiB2, phiB2p, phiB3 = ( np.pi, np.pi, np.pi, np.pi, 0)
        p1 = 1
    elif ans == 'cb1':      #A1,A2,A3,B1,B2,phiA1p,phiB2
        A2 = P[1*j2]*j2
        A3 = P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2)
        B1 = P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3)
        B2 = P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3)
        B3 = 0
        phiA1p = P[-1]
        phiA2, phiA2p, phiA3 = (-phiA1p/2,-phiA1p/2,phiA1p/2)
        phiB1, phiB1p, phiB2, phiB2p, phiB3 = (np.pi, np.pi, 0, 0, 0)
        p1 = 1
    elif ans == 'cb2':      #A1,A2,A3,B1,B2,phiA1p,phiB2
        A2 = P[1*j2]*j2
        A3 = P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2)
        B1 = P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3)
        B2 = P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3)
        B3 = 0
        phiB1 = P[-1]
        phiA1p, phiA2, phiA2p, phiA3 = (0, np.pi, -np.pi, np.pi)        #all can be either 0 or pi except phiA1p
        phiB1p, phiB2, phiB2p, phiB3 = (-phiB1, np.pi, np.pi, 0)
        p1 = 1
    elif ans == 'octa':      #A1,A2,A3,B1,B2,phiA1p,phiB2
        A2 = P[1*j2]*j2
        A3 = 0
        B1 = P[2*j2]*j2 + P[1*(1-j2)]*(1-j2)
        B2 = P[3*j2]*j2
        B3 = P[4*j2*j3]*j2*j3 + P[2*j3*(1-j2)]*j3*(1-j2)
        phiB1 = P[-1]
        phiA1p, phiA2, phiA2p, phiA3 = (np.pi, -np.pi/2, np.pi/2, 0)        #all can be either 0 or pi except phiA1p
        phiB1p, phiB2, phiB2p, phiB3 = (phiB1, 0, 0, -np.pi/2)
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
    N[0,1] = J1*b1p_ *ke1 *t1_  + J2*b2*kf1_
    N[0,2] = J1*b1p  *ke2_*t1   + J2*b2p_*kf2_
    N[0,4] = J1*b1_  *ke1_*t1   + J2*kf1*b2pi
    N[0,5] = J1*b1   *ke2 *t1_  + J2*kf2*b2i_
    N[1,2] = J1*(b1_ *ke3_*t1   + b1p_*ke3*t1_)
    N[1,3] = J1*b1   *ke1 *t1_  + J2*kf1_*b2p_
    N[1,5] = J2*(b2  *kf3_      + b2p*kf3)
    N[2,3] = J1*b1_  *ke2_*t1   + J2*kf2_*b2
    N[2,4] = J2*(b2p_*kf3_      + b2i_*kf3)
    N[3,4] = J1*b1pi_*ke1 *t1_  + J2*b2*kf1_
    N[3,5] = J1*b1p  *ke2_*t1   + J2*b2pi_*kf2_
    N[4,5] = J1*(b1_ *ke3_*t1   + b1pi_*ke3*t1_)
    N[0,0] = J3*kg3_*b3i_*t3_
    N[3,3] = J3*kg3_*b3_ *t3_
    N[1,4] = J3*(b3_*kg2_*t3_  + b3  *kg2 *t3)
    N[2,5] = J3*(b3 *kg1 *t3   + b3i_*kg1_*t3_)
    ####other half square
    N[m+0,m+1] = J1*b1p  *ke1 *t1_ + J2*b2_*kf1_
    N[m+0,m+2] = J1*b1p_ *ke2_*t1  + J2*b2p*kf2_
    N[m+0,m+4] = J1*b1   *ke1_*t1  + J2*kf1*b2pi_
    N[m+0,m+5] = J1*b1_  *ke2 *t1_ + J2*kf2*b2i
    N[m+1,m+2] = J1*(b1  *ke3_*t1  + b1p*ke3*t1_)
    N[m+1,m+3] = J1*b1_  *ke1 *t1_ + J2*kf1_*b2p
    N[m+1,m+5] = J2*(b2_*kf3_ + b2p_*kf3)
    N[m+2,m+3] = J1*b1   *ke2_*t1  + J2*kf2_*b2_
    N[m+2,m+4] = J2*(b2p*kf3_ + b2i*kf3)
    N[m+3,m+4] = J1*b1pi *ke1 *t1_ + J2*b2_*kf1_
    N[m+3,m+5] = J1*b1p_ *ke2_*t1  + J2*b2pi*kf2_
    N[m+4,m+5] = J1*(b1  *ke3_*t1  + b1pi*ke3*t1_)
    N[m+0,m+0] = J3*kg3_*b3i*t3_
    N[m+3,m+3] = J3*kg3_*b3 *t3_
    N[m+1,m+4] = J3*(b3 *kg2_*t3_  + b3_*kg2 *t3)
    N[m+2,m+5] = J3*(b3_*kg1 *t3   + b3i*kg1_*t3_)
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
    N[0,m+1] = - J1*a1p *ke1 *t1_   +J2*a2*kf1_
    N[0,m+2] =   J1*a1p *ke2_*t1    -J2*a2p*kf2_
    N[0,m+4] = - J1*a1  *ke1_*t1    +J2*a2pi*kf1
    N[0,m+5] =   J1*a1  *ke2 *t1_   -J2*a2i*kf2
    N[1,m+2] = - J1*(a1 *ke3_*t1    +a1p*ke3*t1_)
    N[1,m+3] =   J1*a1  *ke1 *t1_   -J2*a2p*kf1_
    N[1,m+5] =   J2*(a2*kf3_  +a2p*kf3)
    N[2,m+3] = - J1*a1  *ke2_*t1    +J2*a2*kf2_
    N[2,m+4] = - J2*(a2p*kf3_ +a2i*kf3)
    N[3,m+4] = - J1*a1pi*ke1 *t1_   +J2*a2*kf1_
    N[3,m+5] =   J1*a1p *ke2_*t1    -J2*a2pi*kf2_
    N[4,m+5] = - J1*(a1 *ke3_*t1    +a1pi*ke3*t1_)
    N[1,m+4] = - J3*(a3*kg2_*t3_  -a3 *kg2 *t3)
    N[2,m+5] =   J3*(a3*kg1 *t3   -a3i*kg1_*t3_)
    N[0,m+0] = - J3*kg3_*a3i*t3_
    N[3,m+3] = - J3*kg3_*a3 *t3_
    #not the diagonal
    N[1,m]   =   J1*a1p *ke1_*t1   -J2*a2*kf1
    N[2,m]   = - J1*a1p *ke2 *t1_  +J2*a2p*kf2
    N[4,m]   =   J1*a1  *ke1 *t1_  -J2*a2pi*kf1_
    N[5,m]   = - J1*a1  *ke2_*t1   +J2*a2i*kf2_
    N[2,m+1] =   J1*(a1 *ke3 *t1_  +a1p*ke3_)
    N[3,m+1] = - J1*a1  *ke1_*t1   +J2*a2p*kf1
    N[5,m+1] = - J2*(a2*kf3   +a2p*kf3_)
    N[3,m+2] =   J1*a1  *ke2 *t1_  -J2*a2*kf2
    N[4,m+2] =   J2*(a2p*kf3  +a2i*kf3_)
    N[4,m+3] =   J1*a1pi*ke1_*t1   -J2*a2*kf1
    N[5,m+3] = - J1*a1p *ke2 *t1_  +J2*a2pi*kf2
    N[5,m+4] =   J1*(a1 *ke3 *t1_  +a1pi*ke3_)
    N[4,m+1] =   J3*(a3*kg2 *t3   -a3 *kg2_*t3_)
    N[5,m+2] = - J3*(a3*kg1_*t3_  -a3i*kg1 *t3)
    N[0,m+0] +=  J3*kg3*a3i*t3
    N[3,m+3] +=  J3*kg3*a3 *t3
    #################################### HERMITIAN MATRIX
    for i in range(2*m):
        for j in range(i,2*m):
            N[j,i] += np.conjugate(N[i,j])
    #################################### L
    for i in range(2*m):
        N[i,i] += L
    return N


