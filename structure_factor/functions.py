import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import cm


def import_data(ans,filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    N = (len(lines)-1)//2 + 1
    P = []
    for i in range(N):
        data = lines[i*2+1].split(',')
        if data[0] != ans:
            continue
        if data[3] != 'True':
            print("Non-converged point, abort")
            exit()
        for d in data[7:]:
            if float(d) != 0.0:
                P.append(float(d))
    return P
####
m = 6
J = np.zeros((2*m,2*m))
for i in range(m):
    J[i,i] = -1
    J[i+m,i+m] = 1

def M(K,P,args):
    m = 6
    N = Nk(K,P,args)
    Ch = LA.cholesky(N) #upper triangular
    w,U = LA.eigh(np.dot(np.dot(Ch,J),np.conjugate(Ch.T)))
    w = np.diag(np.sqrt(np.einsum('ij,j->i',J,w)))
    Mk = np.dot(np.dot(LA.inv(Ch),U),w)
    U,V,X,Y = split(Mk,m,m)
    return U,X,V,Y
####
def split(array, nrows, ncols):
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))
#extended brillouin zone
def EBZ(K):
    x = K[0]
    y = K[1]
    a = np.sqrt(3)
    b = np.pi*8/np.sqrt(3)
    if x < -4*np.pi/3 and (y < -a*x-b or y > a*x+b):
        return False
    if x > 4*np.pi/3 and (y < a*x-b or y > -a*x+b):
        return False
    return True
####
def find_minima(pars,args,Nx,Ny):
    nxg = np.linspace(-1/2,1/2,Nx)
    nyg = np.linspace(-1/2,1/2,Ny)
    K = np.zeros((2,Nx,Ny))
    en = np.zeros((Nx,Ny))
    for i in range(Nx):
        for j in range(Ny):
            K[:,i,j] = np.array([nxg[i]*2*np.pi,(nxg[i]+nyg[j])*2*np.pi/np.sqrt(3)])
            N = Nk(K[:,i,j],pars,args)
            Ch = LA.cholesky(N)
            temp = np.dot(np.dot(Ch,J),np.conjugate(Ch.T))
            en[i,j] = LA.eigvalsh(temp)[m]
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1,projection = '3d')
    #ax.plot_surface(K[0],K[1],en,cmap=cm.plasma)
    #plt.show()
    #exit()
    ind1 = np.argmin(en)
    k1 = K[:,ind1//Nx,ind1%Ny]
    en[ind1//Nx,ind1%Ny] += 10
    ind2 = np.argmin(en)
    k2 = K[:,ind2//Nx,ind2%Ny]
    if np.abs(np.abs(k1[0]-k2[0])-2*np.pi) < 1e-3 or np.abs(np.abs(k1[1]-k2[1])-2*np.pi/np.sqrt(3)) < 1e-3:
        en[ind2//Nx,ind2%Ny] += 10
        ind3 = np.argmin(en)
        k2 = K[:,ind3//Nx,ind3%Ny]
        en[ind2//Nx,ind2%Ny] -= 10
    en[ind1//Nx,ind1%Ny] -= 10
    if LA.norm(k1-k2) < 4*np.pi/np.sqrt(3)/(Ny-2):
        K_ = [k1]
    else: 
        K_ = [k1,k2]
    plt.figure()
    plt.scatter(K[0],K[1],c=en,cmap = cm.plasma)
    plt.colorbar()
    for k in K_:
        plt.scatter(k[0],k[1],c='r',marker='*')
        print(k)
    plt.show()
    ok = input("Is it ok?[y/n] ([1] for keeping only first value found]\t")
    if ok == 'n':
        exit()
    if ok == '1':
        K_ = [k1]
    LRO = True if en[ind1//Nx,ind1%Ny] < 0.05 else False
    return K_, LRO
####
def get_V(K_,pars,args):
    V = []
    for K in K_:
        N = Nk(K,pars,args)
        Ch = LA.cholesky(N) #upper triangular
        w,U = LA.eigh(np.dot(np.dot(Ch,J),np.conjugate(Ch.T)))
        w_ = np.diag(np.sqrt(np.einsum('ij,j->i',J,w)))
        Mk = np.dot(np.dot(LA.inv(Ch),U),w_)
        V.append(Mk[:,m-1])
        if np.abs(w[m]-w[m+1]) < 1e-3:
            print("degenerate K point")
            V.append(Mk[:,m+1])
    return V
####
f = np.sqrt(3)/4
def SpinStructureFactor(k,L,UC):
    a1 = np.array([1,0])
    a2 = np.array([-1,np.sqrt(3)])
    d = np.array([  [1/2,-1/4,1/4,0,-3/4,-1/4],
                    [0,f,f,2*f,3*f,3*f]])
    resxy = 0
    resz = 0
    dist = np.zeros(2)
    for i in range(UC//2,UC//2+1):
        for j in range(UC//2,UC//2+1):
            for l in range(6):
#                Li = L[:,l+j%2*3,i,j//2*(j+1)%2 + (j-1)*j%2]
                Li = L[:,l,i,j]
                ri = i*a1+j*a2+d[:,l]
                for i2 in range(UC):
                    for j2 in range(UC):
                        for l2 in range(6):
#                            Lj = L[:,l2+j2%2*3,i2,j2//2*(j2+1)%2 + (j2-1)*j2%2]
                            Lj = L[:,l2,i2,j2]
                            rj = i2*a1+j2*a2+d[:,l2]
                            dist = ri-rj
                            SiSjxy = Li[0]*Lj[0] + Li[1]*Lj[1]#np.dot(Li,L[i2,j2,l2])
                            SiSjz = Li[2]*Lj[2]
                            resxy += np.cos(np.dot(k,dist))*SiSjxy/2
                            resz += np.cos(np.dot(k,dist))*SiSjz/2
    #print(resxy,resz)
    return np.real(resxy), np.real(resz)
####
def Nk(K,par,args):
    a1 = (1,0)
    a2 = (-1,np.sqrt(3))
    a12p = (a1[0]+a2[0],a1[1]+a2[1])
    a12m = (a1[0]-a2[0],a1[1]-a2[1])
    ka1 = np.exp(1j*np.dot(a1,K));   ka1_ = np.conjugate(ka1);
    ka2 = np.exp(1j*np.dot(a2,K));   ka2_ = np.conjugate(ka2);
    ka12p = np.exp(1j*np.dot(a12p,K));   ka12p_ = np.conjugate(ka12p);
    ka12m = np.exp(1j*np.dot(a12m,K));   ka12m_ = np.conjugate(ka12m);
    J1,J2,J3,ans,DM = args
    DM1 = 0
    DM3 = 0
    if DM:
        DM1 = 0#4*np.pi/3
        DM3 = 0#2*np.pi/3
    t1 = np.exp(-1j*DM1/2);    t1_ = np.conjugate(t1)
    t3 = np.exp(-1j*DM3/2);    t3_ = np.conjugate(t3)
    m = 6
    L = par[0]
    P = par[1:]
    J1 /= 2.
    J2 /= 2.
    J3 /= 2.
    j2 = np.sign(int(np.abs(J2)*1e8))   #check if it is 0 or not --> problem with VERY small J2,J3
    j3 = np.sign(int(np.abs(J3)*1e8))
    A1 = P[0]
    #params
    if ans == '3x3_1' and DM:
        A2 = 0;     phiA2 = 0;    phiA2p = 0;
        A3 = P[1*j3]*j3
        B1 = P[2*j3]*j3 + P[1]*(1-j3)
        B2 = P[3*j2*j3]*j2*j3+P[2*j2*(1-j3)]*(1-j3)*j2
        B3 = P[4*j3*j2]*j3*j2+P[3*j3*(1-j2)]*j3*(1-j2)
        phiB1 = P[-3*j3]*j3*j2 + P[-2]*j3*(1-j2) + P[-2]*j2*(1-j3) + P[-1]*(1-j2)*(1-j3)
        phiB2 = P[-2]*j2*j3 + P[-1]*j2*(1-j3)
        phiA3 = P[-1]*j3
        phiA1p = np.pi
        phiB1p, phiB2p, phiB3 = (phiB1, phiB2, np.pi)
        p1 = 0
    elif ans == '3x3_1' and not DM:
        A2 = 0;     phiA2 = 0;    phiA2p = 0;
        A3 = j3*P[1*j3]
        B1 = P[2*j3]*j3 + P[1]*(1-j3)
        B2 = P[3*j2*j3]*j2*j3+P[2*j2*(1-j3)]*(1-j3)*j2
        B3 = P[4*j3*j2]*j3*j2+P[3*j3*(1-j2)]*j3*(1-j2)
        phiA3 = P[-1]*j3
        phiA1p = np.pi
        phiB1, phiB1p, phiB2, phiB2p, phiB3 = (np.pi, np.pi, 0, 0, np.pi)
        p1 = 0
    elif ans == 'q0_1' and DM:
        A3 = 0; phiA3 = 0
        A2 = P[1]*j2
        B1 = P[2*j2]*j2+P[1]*(1-j2)
        B2 = P[3*j2]*j2
        B3 = P[4*j3*j2]*j3*j2+P[2*j3*(1-j2)]*j3*(1-j2)
        phiB1 = P[-3*j2]*j2 + P[-1]*(1-j2)
        phiA2 = P[-2]*j2
        phiB2 = P[-1]*j2
        phiA1p, phiA2p = (0, phiA2)
        phiB1p, phiB2p, phiB3 = (phiB1, phiB2, 0)
        p1 = 0
    elif ans == 'q0_1' and not DM:
        A3 = 0; phiA3 = 0
        A2 = P[1]*j2
        B1 = P[2*j2]*j2+P[1]*(1-j2)
        B2 = P[3*j2]*j2
        B3 = P[4*j3*j2]*j3*j2+P[2*j3*(1-j2)]*j3*(1-j2)
        phiA2 = P[-1]*j2
        phiA1p, phiA2p = (0, phiA2)
        phiB1, phiB1p, phiB2, phiB2p, phiB3 = (np.pi, np.pi, np.pi, np.pi, 0)
        p1 = 0
    elif ans == 'cb1' and DM:      #A1,A2,A3,B1,B2,phiA1p,phiB2
        B3 = 0; phiB3 = 0
        A2 = P[1*j2]*j2
        A3 = P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2)
        B1 = P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3)
        B2 = P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3)
        phiA1p = P[-4*j2]*j2 + P[-2]*(1-j2)
        phiB1 = P[-3*j2]*j2 + P[-1]*(1-j2)
        phiA2 = P[-2]*j2
        phiB2 = P[-1]*j2
        phiA2p, phiA3 = (phiA1p-phiA2,phiA1p/2)
        phiB1p, phiB2p= (-phiB1 ,-phiB2)
        p1 = 1
    elif ans == 'cb1' and not DM:      #A1,A2,A3,B1,B2,phiA1p,phiB2
        B3 = 0; phiB3 = 0
        A2 = P[1*j2]*j2
        A3 = P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2)
        B1 = P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3)
        B2 = P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3)
        phiA1p = P[-2]*j2 + P[-1]*(1-j2)
        phiB2 = P[-1]*j2
        phiA2, phiA2p, phiA3 = (np.pi+phiA1p/2,np.pi+phiA1p/2,phiA1p/2)
        phiB1, phiB1p, phiB2p= (np.pi, np.pi, -phiB2)
        p1 = 1
    ################
    N = np.zeros((2*m,2*m), dtype=complex)
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
    #
    N[0,1] = J1*b1p_ *ka1  *t1_              + J2*b2
    N[0,2] = J1*b1p        *t1               + J2*b2p_ *ka1
    N[0,4] = J1*b1_  *ka2_ *t1               + J2*b2pi *ka12m
    N[0,5] = J1*b1   *ka2_ *t1_              + J2*b2i_ *ka12p_
    N[1,2] = J1*(b1_       *t1  + b1p_*ka1_*t1_)
    N[1,3] = J1*b1         *t1_              + J2*b2p_ *ka1_
    N[1,5] =                                   J2*(b2  *ka12p_ + b2p)
    N[2,3] = J1*b1_        *t1               + J2*b2   *ka1
    N[2,4] =                                   J2*(b2p_*ka2_ + b2i_*ka1)
    N[3,4] = J1*b1pi_*ka1  *t1_              + J2*b2
    N[3,5] = J1*b1p        *t1               + J2*b2pi_*ka1
    N[4,5] = J1*(b1_       *t1  + b1pi_*ka1_*t1_)

    N[0,0] = J3*b3i_ *ka1_ *t3_
    N[3,3] = J3*b3_  *ka1_ *t3_
    N[1,4] = J3*(b3_ *ka2_ *t3_ + b3       *t3)
    N[2,5] = J3*(b3  *ka12p_  *t3  + b3i_*ka1*t3_)
    ####other half square
    N[m+0,m+1] = J1*b1p  *ka1  *t1_           + J2*b2_
    N[m+0,m+2] = J1*b1p_       *t1            + J2*b2p  *ka1
    N[m+0,m+4] = J1*b1   *ka2_ *t1            + J2*b2pi_*ka12m
    N[m+0,m+5] = J1*b1_  *ka2_ *t1_           + J2*b2i  *ka12p_
    N[m+1,m+2] = J1*(b1        *t1  + b1p*ka1_*t1_)
    N[m+1,m+3] = J1*b1_        *t1_           + J2*b2p  *ka1_
    N[m+1,m+5] =                                J2*(b2_ *ka12p_ + b2p_)
    N[m+2,m+3] = J1*b1         *t1            + J2*b2_  *ka1
    N[m+2,m+4] =                                J2*(b2p *ka2_ + b2i *ka1)
    N[m+3,m+4] = J1*b1pi *ka1  *t1_           + J2*b2_
    N[m+3,m+5] = J1*b1p_       *t1            + J2*b2pi *ka1
    N[m+4,m+5] = J1*(b1        *t1  + b1pi*ka1_*t1_)

    N[m+0,m+0] = J3*b3i *ka1_ *t3_
    N[m+3,m+3] = J3*b3  *ka1_ *t3_
    N[m+1,m+4] = J3*(b3 *ka2_ *t3_ + b3_  *t3)
    N[m+2,m+5] = J3*(b3_*ka12p_  *t3  + b3i *ka1*t3_)
    ######################################## A
    a1 =    A1
    a1p =   A1*np.exp(1j*phiA1p)
    a1pi =  A1*np.exp(1j*(phiA1p+p1*np.pi))
    a2 =    A2*np.exp(1j*phiA2)
    a2i =   A2*np.exp(1j*(phiA2+p1*np.pi))
    a2p =   A2*np.exp(1j*phiA2p)
    a2pi =  A2*np.exp(1j*(phiA2p+p1*np.pi))
    a3 =    A3*np.exp(1j*phiA3)
    a3i =   A3*np.exp(1j*(phiA3+p1*np.pi))
    N[0,m+1] = - J1*a1p *ka1 *t1_           +J2*a2
    N[0,m+2] =   J1*a1p      *t1            -J2*a2p  *ka1
    N[0,m+4] = - J1*a1  *ka2_*t1            +J2*a2pi *ka12m
    N[0,m+5] =   J1*a1  *ka2_*t1_           -J2*a2i  *ka12p_
    N[1,m+2] = - J1*(a1      *t1   +a1p*ka1_*t1_)
    N[1,m+3] =   J1*a1       *t1_           -J2*a2p  *ka1_
    N[1,m+5] =                               J2*(a2  *ka12p_  +a2p)
    N[2,m+3] = - J1*a1       *t1            +J2*a2   *ka1
    N[2,m+4] =                              -J2*(a2p *ka2_  +a2i*ka1)
    N[3,m+4] = - J1*a1pi*ka1 *t1_           +J2*a2
    N[3,m+5] =   J1*a1p      *t1            -J2*a2pi *ka1
    N[4,m+5] = - J1*(a1      *t1   +a1pi*ka1_*t1_)

    N[0,m+0] = - J3*a3i *ka1_*t3_
    N[3,m+3] = - J3*a3  *ka1_*t3_
    N[1,m+4] = - J3*(a3 *ka2_*t3_  -a3 *t3)
    N[2,m+5] = - J3*(a3i*ka1*t3_  -a3 *ka12p_ *t3)
    #not the diagonal
    N[1,m]   =   J1*a1p *ka1_*t1            -J2*a2
    N[2,m]   = - J1*a1p      *t1_           +J2*a2p  *ka1_
    N[4,m]   =   J1*a1  *ka2 *t1_           -J2*a2pi *ka12m_
    N[5,m]   = - J1*a1  *ka2 *t1            +J2*a2i  *ka12p
    N[2,m+1] =   J1*(a1      *t1_  +a1p*ka1 *t1)
    N[3,m+1] = - J1*a1       *t1            +J2*a2p  *ka1_
    N[5,m+1] =                              -J2*(a2  *ka12p   +a2p)
    N[3,m+2] =   J1*a1       *t1_           -J2*a2   *ka1_
    N[4,m+2] =                               J2*(a2p *ka2   +a2i*ka1_)
    N[4,m+3] =   J1*a1pi*ka1_*t1            -J2*a2
    N[5,m+3] = - J1*a1p      *t1_           +J2*a2pi *ka1_
    N[5,m+4] =   J1*(a1      *t1_  +a1pi*ka1 *t1)

    N[0,m+0] +=  J3*a3i *ka1  *t3
    N[3,m+3] +=  J3*a3  *ka1  *t3
    N[4,m+1] =   J3*(a3 *ka2 *t3   -a3 *t3_)
    N[5,m+2] =   J3*(a3i*ka1_ *t3   -a3 *ka12p *t3_)
    #################################### HERMITIAN MATRIX
    #N += np.conjugate(N.transpose((1,0,2,3)))
    for i in range(2*m):
        for j in range(i,2*m):
            N[j,i] += np.conjugate(N[i,j])
    #################################### L
    for i in range(2*m):
        N[i,i] += L
    return N


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

