import numpy as np
from scipy import linalg as LA
from scipy.integrate import dblquad
import inputs_2nn as inp

######## Generate alpha pts
def generate_alpha(params,kp,dp):
    ans = params[2]
    J2 = params[1]
    K1 = np.linspace(0,inp.maxK1[ans],kp)  #Kx in BZ
    K2 = np.linspace(0,inp.maxK2[ans],kp)  #Ky in BZ
    xdata = np.linspace(-0.5,np.pi*2+0.5,dp)
    ydata = np.ndarray(dp)
    for i in range(dp):
        ydata[i] = np.sqrt(np.amax(eigG2_arr(K1,K2,params,xdata[i]).ravel()))
    data = np.ndarray((2,dp))
    data[0] = xdata
    data[1] = ydata
    np.save('alphaD/alphas_kp='+str(kp)+'dp='+str(dp)+'J2='+str(J2).replace('.',',')+'.npy',data)
######## Find bounds
def bounds_alpha(ratio,interp):
    conv_tol = 1e-4
    minalpha = np.pi/4
    maxalpha = np.pi/2
    pts = 1000
    range_al = np.linspace(np.pi/4,np.pi/2,pts)
    for a in range(1,pts):
        temp = interp(range_al[a])
        if np.abs(temp+ratio) < conv_tol and interp(range_al[a-1])-temp < 0:
            maxalpha = range_al[a]
            break
    return maxalpha
########## SUMS
#Function which returns e^(-i(k \dot a)) with k the 
#   momentum and a the position vector. m is the unit
#   cell size in real space, which can be 3 or 6.
#   a1 points to the right and a2 to up-left.
def exp_k(kx,ky,a1,a2,m):
    ax = a1+a2*(-1/2)*m/3
    ay = a2*(np.sqrt(3)/2)*m/3
    res = np.zeros((len(kx),len(ky)),dtype=complex)
    for i,I in enumerate(kx):
        for j,J in enumerate(ky):
            res[i,j] = np.exp(-1j*(I*ax+J*ay)/2)
    return res
#Evaluates the eigenvalues of (G^dag G) for the four different
#   ansatze and for arrays of momenta k1,k2 in the BZ. 
#   Params contains the couplings J1, J2, the DM phase phi 
#   and the ansatz considered.
def eigG2_arr(k1, k2, params, alpha):
    J1,J2,phi1,ans = params
    #phi1 = inp.phi1#np.pi/3*2
    phi2 = inp.phi2#np.pi
    m = inp.m[ans]
    eta1 = np.exp(-1j*phi1)
    eta1_ = np.conjugate(eta1)
    eta2 = np.exp(-1j*phi2)
    eta2_ = np.conjugate(eta2)
    G = np.zeros((m,m,len(k1),len(k2)),dtype=complex)
    #1nn
    if m == 3:
        G[0,1] = np.sin(alpha)*(-J1*(eta1_*(-1)**(ans)*exp_k(k1,k2,0,-1,m)-eta1_*exp_k(k1,k2,0,1,m)))
        G[0,1] += np.cos(alpha)*(-J2*(-eta2_*exp_k(k1,k2,2,1,m)-eta2_*exp_k(k1,k2,-2,-1,m)))
        G[0,2] = np.sin(alpha)*(-J1*(eta1*exp_k(k1,k2,1,1,m)-eta1*(-1)**(ans)*exp_k(k1,k2,-1,-1,m)))
        G[0,2] += np.cos(alpha)*(-J2*(eta2*exp_k(k1,k2,-1,1,m)+eta2*exp_k(k1,k2,1,-1,m)))
        G[1,2] = np.sin(alpha)*(-J1*(-eta1_*exp_k(k1,k2,1,0,m)+eta1_*(-1)**(ans)*exp_k(k1,k2,-1,0,m)))
        G[1,2] += np.cos(alpha)*(-J2*(-eta2_*exp_k(k1,k2,1,2,m)-eta2_*exp_k(k1,k2,-1,-2,m)))
    elif m == 6:
        G[0,1] = -J1*eta_*(-1)**(ans-2)*exp_k(k1,k2,0,-1,m)
        G[0,2] = -J1*eta*(-(-1)**(ans-2)*exp_k(k1,k2,-1,-1,m))
        G[0,4] = -J1*eta_*(-exp_k(k1,k2,0,1,m))
        G[0,5] = -J1*eta*exp_k(k1,k2,1,1,m)
        G[1,2] = -J1*eta_*(-exp_k(k1,k2,1,0,m)+(-1)**(ans-2)*exp_k(k1,k2,-1,0,m))
        G[1,3] = -J1*eta*exp_k(k1,k2,0,-1,m)
        G[2,3] = -J1*eta_*(-1)*exp_k(k1,k2,-1,-1,m)
        G[3,4] = -J1*eta_*(-1)**(ans-2)*exp_k(k1,k2,0,-1,m)
        G[3,5] = -J1*eta*exp_k(k1,k2,-1,-1,m)*(-1)**(ans-2)
        G[4,5] = -J1*eta_*(-exp_k(k1,k2,1,0,m)-(-1)**(ans-2)*exp_k(k1,k2,-1,0,m))
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

def sum_lam(ratio, ans, g2):
    m = inp.m[ans]
    norm = len(g2.ravel())
    res = (ratio/np.sqrt(ratio**2-g2)).ravel().sum()
    res /= norm
    return res

def sum_mf(ratio, ans, g2):
    m = inp.m[ans]
    norm = len(g2.ravel())
    res = (np.sqrt(ratio**2-g2)).ravel().sum()
    res /= norm
    return res

