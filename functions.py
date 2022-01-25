import numpy as np
from numpy import linalg as LA
from scipy.integrate import dblquad
import input_variables as inp

#exp K
def exp_k(kx,ky,a1,a2,m):
    ax = a1+a2*(-1/2)*m/3
    ay = a2*(np.sqrt(3)/2)*m/3
    res = np.zeros((len(kx),len(ky)),dtype=complex)
    for i,I in enumerate(kx):
        for j,J in enumerate(ky):
            res[i,j] = np.exp(-1j*(I*ax+J*ay))
    return res
def exp_k2(kx,ky,a1,a2,m):
    ax = a1+a2*(-1/2)*m/3
    ay = a2*(1/2*np.sqrt(3))*m/3
    res = np.exp(-1j*(kx*ax+ky*ay))
    return res

#g^2 calculators
def eigG2_arr(k1, k2, params):
#    if np.isscalar(k1) and np.isscalar(k2):
#        return eigG2(k1,k2,unit)
    J1,J2,phi,ans = params
    m = inp.m[ans]
    eta = np.exp(-1j*phi)
    eta_ = np.conjugate(eta)
    G = np.zeros((m,m,len(k1),len(k2)),dtype=complex)
    #1nn
    if m == 3:
        G[0,1] = -J1*eta*((-1)**ans-exp_k(k1,k2,0,1,m))
        G[0,2] = -J1*eta*(exp_k(k1,k2,0,1,m)-(-1)**ans*exp_k(k1,k2,-1,0,m))
        G[1,2] = -J1*eta*((-1)**ans*exp_k(k1,k2,-1,0,m)-1)
        #G[0,1] = -J1*(eta*(-1)**ans-eta_*exp_k(k1,k2,0,1,m))
        #G[0,2] = -J1*(eta*exp_k(k1,k2,0,1,m)-eta_*(-1)**ans*exp_k(k1,k2,-1,0,m))
        #G[1,2] = -J1*(-eta+eta_*(-1)**ans*exp_k(k1,k2,-1,0,m))
    elif m == 6:
        G[0,1] = -J1*eta*(-1)**(ans-2)
        G[0,2] = -J1*eta_*(-(-1)**(ans-2)*exp_k(k1,k2,-1,0,m))
        G[0,4] = -J1*eta_*(-exp_k(k1,k2,0,1,m))
        G[0,5] = -J1*eta*(exp_k(k1,k2,0,1,m))
        G[1,2] = -J1*(eta_*(-1)+eta*(-1)**(ans-2)*exp_k(k1,k2,-1,0,m))
        G[1,3] = -J1*eta        ###sure?
        G[2,3] = -J1*eta_*(-1)
        G[3,4] = -J1*eta*(-1)**(ans-2)
        G[3,5] = -J1*eta_*exp_k(k1,k2,-1,0,m)*(-1)**(ans-2)
        G[4,5] = -J1*(eta_*(-1)-eta*(-1)**(ans-2)*exp_k(k1,k2,-1,0,m))
    #c.c.
    #G = G + G.T
    res = np.zeros((m,len(k1),len(k2)))
    for i in range(len(k1)):
        for j in range(len(k2)):
            G[:,:,i,j] = G[:,:,i,j] - np.conjugate(G[:,:,i,j].T)
            G_ = np.conjugate(G[:,:,i,j]).T
            GG_ = np.matmul(G[:,:,i,j],G_)
            res[:,i,j] = LA.eigvalsh(GG_)
    return np.absolute(res)

########## SUMS
def sum_en(x, params):
    m = inp.m[params[3]]
    max1 = inp.maxK1[params[3]]
    max2 = inp.maxK2[params[3]]
    K1 = np.linspace(0,max1,inp.sum_pts)
    K2 = np.linspace(0,max2,inp.sum_pts)
    res = np.sqrt(x[1]**2-x[0]**2*eigG2_arr(K1,K2,params)).ravel().sum()
    res /= (inp.sum_pts**2)
    return res/m

def sum_lam(ratio,params):
    m = inp.m[params[3]]
    max1 = inp.maxK1[params[3]]
    max2 = inp.maxK2[params[3]]
    K1 = np.linspace(0,max1,inp.sum_pts)
    K2 = np.linspace(0,max2,inp.sum_pts)
    res = (ratio/np.sqrt(ratio**2-eigG2_arr(K1,K2,params))).ravel().sum()
    res /= (inp.sum_pts**2)
    return res/m

def sum_mf(ratio,params):
    m = inp.m[params[3]]
    max1 = inp.maxK1[params[3]]
    max2 = inp.maxK2[params[3]]
    K1 = np.linspace(0,max1,inp.sum_pts)
    K2 = np.linspace(0,max2,inp.sum_pts)
    res = (np.sqrt(ratio**2-eigG2_arr(K1,K2,params))).ravel().sum()
    res /= (inp.sum_pts**2)
    return res/m

########## INTEGRALS
def eigG2(k1, k2, params):
    J1, J2, phi, ans = params
    m = inp.m[ans]
    eta = np.exp(-1j*phi)
    G = np.zeros((m,m),dtype=complex)
    #1nn
    if m == 3:
        G[0,1] = -J1*eta*((-1)**ans-exp_k2(k1,k2,0,1,m))
        G[0,2] = -J1*eta*(exp_k2(k1,k2,0,1,m)-(-1)**ans*exp_k2(k1,k2,-1,0,m))
        G[1,2] = -J1*eta*((-1)**ans*exp_k2(k1,k2,-1,0,m)-1)
    elif m == 6:
        G[0,1] = -J1*eta*(-1)**(ans-2)
        G[0,2] = -J1*eta*(-exp_k2(k1,k2,-1,0,m))
        G[0,4] = -J1*eta*(-exp_k2(k1,k2,0,1,m))
        G[0,5] = -J1*eta*(exp_k2(k1,k2,0,1,m))
        G[1,2] = -J1*eta*(-1+exp_k2(k1,k2,-1,0,m))
        G[1,3] = -J1*eta
        G[2,3] = -J1*eta*(-1)
        G[3,4] = -J1*eta*(-1)**(ans-2)
        G[3,5] = -J1*eta*(exp_k2(k1,k2,-1,0,m))
        G[4,5] = -J1*eta*(-1-exp_k2(k1,k2,-1,0,m))
    #c.c.
    res = np.zeros(m)
    G = G - np.conjugate(G).T
    G_ = np.conjugate(G).T
    GG_ = np.matmul(G,G_)
    res = LA.eigvalsh(GG_)
    return np.absolute(res)

def int_en(x,params):
    m = inp.m[params[3]]
    max1 = inp.maxK1[params[3]]
    max2 = inp.maxK2[params[3]]
    res = dblquad(
            lambda k1,k2:(np.sqrt(x[1]**2-x[0]**2*eigG2(k1,k2,params)).sum()),
            0,max1,
            lambda k2: 0,
            lambda k2: max2
            )[0]
    res /= (max1*max2)
    return res/m

def int_lam(ratio,params):
    m = inp.m[params[3]]
    max1 = inp.maxK1[params[3]]
    max2 = inp.maxK2[params[3]]
    res = dblquad(
            lambda k1,k2:(ratio/np.sqrt(ratio**2-eigG2(k1,k2,params))).sum(),
            0,max1,
            lambda k2: 0,
            lambda k2: max2
            )[0]
    res /= (max1*max2)
    return res/m

def int_mf(ratio,params):
    m = inp.m[params[3]]
    max1 = inp.maxK1[params[3]]
    max2 = inp.maxK2[params[3]]
    res = dblquad(
            lambda k1,k2:(np.sqrt(ratio**2-eigG2(k1,k2,params)).sum()),
            0,max1,
            lambda k2: 0,
            lambda k2: max2
            )[0]
    res /= (max1*max2)
    return res/m

