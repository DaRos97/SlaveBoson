import numpy as np
from scipy import linalg as LA
from scipy.integrate import dblquad
import inputs_2nn as inp

######## Generate alpha pts
def generate_alpha(params,kp,dp,path_alpha):
    ans = params[3]
    K1 = np.linspace(0,inp.maxK1[ans],kp)  #Kx in BZ
    K2 = np.linspace(0,inp.maxK2[ans],kp)  #Ky in BZ
    xdata = np.linspace(inp.alpha_min,inp.alpha_max,dp)
    ydata = np.ndarray(dp)
    for i in range(dp):
        ydata[i] = np.sqrt(np.amax(eigG2(K1,K2,params,xdata[i]).ravel()))
    data = np.ndarray((2,dp))
    data[0] = xdata
    data[1] = ydata
    np.save(path_alpha,data)
#####
def inv_interpolate(data,ratio):
    # Initialize final x
    y = -ratio
    x = 0
    n = len(data[0])
    for i in range(n):
        # Calculate each term
        # of the given formula
        xi = data[0,i]
        for j in range(n):
            if j != i:
                xi = (xi * (y - data[1,j]) /
                      (data[1,i] - data[1,j]))
        # Add term to final result
        x += xi
    return x
######### Find bounds
def bnd_alpha(data,ratio):
    xdata = data[0]
    ydata = data[1]
    y = -ratio
    y_ = np.ndarray(len(ydata))
    for i in range(len(ydata)):
        y_[i] = np.abs(y-ydata[i])
    min_pt = np.argmin(y_)
    if np.abs(y-ydata[min_pt]) > 0.1:
        return 0
    if y-ydata[min_pt] > 0:
        b = [xdata[min_pt-1],ydata[min_pt-1]]
        a = [xdata[min_pt],ydata[min_pt]]
    else:
        b = [xdata[min_pt],ydata[min_pt]]
        a = [xdata[min_pt+1],ydata[min_pt+1]]
    k = (a[1]-b[1])/(a[0]-b[0])
    c = b[1]-b[0]*k
    xfin = (y-c)/k
    if xfin+0.001<np.pi/4:
        xfin+= 0.001
    return xfin
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
            res[i,j] = np.exp(-1j*(I*ax+J*ay))
    return res
#Evaluates the eigenvalues of (G^dag G) for the four different
#   ansatze and for arrays of momenta k1,k2 in the BZ. 
#   Params contains the couplings J1, J2, the DM phase phi 
#   and the ansatz considered.
def eigG2(k1, k2, params, alpha):
    J1,J2,J3e,ans = params
    phi1 = inp.phi1#np.pi/3
    phi2 = inp.phi2#np.pi
    phi3 = inp.phi3#
    m = inp.m[ans]
    eta1 = np.exp(-1j*phi1)
    eta1_ = np.conjugate(eta1)
    eta2 = np.exp(-1j*phi2)
    eta2_ = np.conjugate(eta2)
    eta3 = np.exp(-1j*phi3)
    eta3_ = np.conjugate(eta3)
    G = np.zeros((m,m,len(k1),len(k2)),dtype=complex)
    #1nn
    if m == 3:
        G[0,1] = np.cos(alpha)*(-2*J1*(eta1_*(-1)**(ans)-eta1_*exp_k(k1,k2,0,1,m)))
        G[0,1] += ans*np.sin(alpha)*(-2*J2*(-eta2_*exp_k(k1,k2,1,1,m)-eta2_*exp_k(k1,k2,-1,0,m)))
        G[0,2] = np.cos(alpha)*(-2*J1*(eta1*exp_k(k1,k2,0,1,m)-eta1*(-1)**(ans)*exp_k(k1,k2,-1,0,m)))
        G[0,2] += ans*np.sin(alpha)*(-2*J2*(eta2+eta2*exp_k(k1,k2,-1,1,m)))
        G[1,2] = np.cos(alpha)*(-2*J1*(-eta1_+eta1_*(-1)**(ans)*exp_k(k1,k2,-1,0,m)))
        G[1,2] += ans*np.sin(alpha)*(-2*J2*(-eta2_*exp_k(k1,k2,0,1,m)-eta2_*exp_k(k1,k2,-1,-1,m)))
        G[0,0] = (1-ans)*np.sin(alpha)*(-2*J3e*eta3*exp_k(k1,k2,1,0,m))
        G[1,1] = (1-ans)*np.sin(alpha)*(-2*J3e*eta3*exp_k(k1,k2,1,1,m))
        G[2,2] = (1-ans)*np.sin(alpha)*(-2*J3e*eta3*exp_k(k1,k2,0,1,m))
    else:
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

def sum_xi(params,alpha,ratio,g2):
    J1,J2,J3,ans = params
    z1 = inp.z1
    z2 = inp.z2
    z3 = inp.z3
    norm = len(g2.ravel())
    Sum = (g2/np.sqrt(ratio**2-g2)).ravel().sum()
    res = Sum/(2*(2*(z1*J1*np.cos(alpha)**2+ans*z2*J2*np.sin(alpha)**2+(1-ans)*z3*J3*np.sin(alpha)**2)))
    res /= norm
    return res
