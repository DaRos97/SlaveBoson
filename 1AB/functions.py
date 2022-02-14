import numpy as np
from scipy import linalg as LA
from scipy.integrate import dblquad
from scipy.interpolate import interp1d
import inputs as inp
import time

#Some parameters from inputs.py
kp = inp.sum_pts
k = (inp.K1,inp.K2)
S = inp.S
J1 = inp.J1

#Function which returns e^(-i(k \dot a)) with k the 
#   momentum and a the position vector. m is the unit
#   cell size in real space, which can be 3 or 6.
#   a1 points to the right and a2 to up-left.
def exp_k(k,a1,a2):
    ax = a1+a2*(-1/2)
    ay = a2*(np.sqrt(3)/2)
    res = np.ndarray((kp,kp),dtype=complex)
    for i in range(kp):
        for j in range(kp):
            res[i,j] = np.exp(-1j*(k[0][i]*ax+k[1][j]*ay))
    return res
#Evaluates the eigenvalues of (G^dag G) for the four different
#   ansatze and for arrays of momenta k1,k2 in the BZ. 
#   Params contains the couplings J1, J2, the DM phase phi 
#   and theansatz considered.
def eigN(P):
    A = P[0]
    B = P[1]
    L = P[2]
    N = np.zeros((6,6,kp,kp),dtype=complex)
    N[0,1] = -J1/2*B*(1+exp_k(k,0,1))
    N[0,2] = -J1/2*B*(exp_k(k,0,1)+exp_k(k,-1,0))
    N[1,2] = -J1/2*B*(1+exp_k(k,-1,0))
    N[3,4] = N[0,1]
    N[3,5] = N[0,2]
    N[4,5] = N[1,2]
    N[0,4] = J1/2*A*(exp_k(k,0,1)-1)
    N[0,5] = J1/2*A*(exp_k(k,-1,0)-exp_k(k,0,1))
    N[1,5] = J1/2*A*(1-exp_k(k,-1,0))
    N[1,3] = -np.conjugate(N[0,4])
    N[2,3] = -np.conjugate(N[0,5])
    N[2,4] = -np.conjugate(N[1,5])
    res = np.zeros((3,kp,kp))
    for i in range(kp):
        for j in range(kp):
            N[:,:,i,j] = N[:,:,i,j] + np.conjugate(N[:,:,i,j]).T
            for l in range(6):
                N[l,l,i,j] = L
            for m in range(3,6):
                for n in range(6):
                    N[m,n,i,j] *= -1
            temp = LA.eigvals(N[:,:,i,j])
            res[:,i,j] = np.sort(temp.real)[3:]             #problem of imaginary part -> is not 0
    del N
    return res

def tot_E(A,B,L):
    eN = eigN([A,B,L])
    E2 = 2*inp.J1*(A**2-B**2)-L*(2*S+1)
    res = eN.ravel().sum()/(3*kp*kp) + E2
    return res

def Sigmaa(A,B,L):
    pts = inp.derPts
    res = 0
    p = [A,B,L]
    k = [4*p[0]*J1,-4*p[1]*J1,-2*S-1]
    ran = inp.ran
    e = np.ndarray(pts)
    for i in range(len(p)):
        rangeP = np.linspace(p[i]-ran[i],p[i]+ran[i],pts)
        pp = p
        for j in range(pts):
            pp[i] = rangeP[j]
            e[j] = eigN(pp).ravel().sum()/(3*kp**2)
        de = np.gradient(e)
        dx = np.gradient(rangeP)
        der = de/dx
        f = interp1d(rangeP,der)
        res += (k[i]+f(p[i]))**2
    return res




