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
def eigN(P,args):
    A1 = P[0]
    A2 = P[1]
    A3 = P[2]
    L = P[3]
    J1,J2,J3 = args
    N = np.zeros((6,6,kp,kp),dtype=complex)
    N[0,4] = J1/2*A1*(exp_k(k,0,1)-1)
    N[0,5] = J1/2*A1*(exp_k(k,-1,0)-exp_k(k,0,1))
    N[1,5] = J1/2*A1*(1-exp_k(k,-1,0))
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

def Sigma(P,args):
    res = 0
    J1,J2,J3 = args
    k = [4*P[0]*J1,4*P[1]*J2,2*P[2]*J3,-2*S-1]
    ran = inp.der_range
    e = np.ndarray(inp.der_pts)
    for i in range(len(P)):
        rangeP = np.linspace(P[i]-ran[i],P[i]+ran[i],inp.der_pts)
        pp = P
        for j in range(inp.der_pts):
            pp[i] = rangeP[j]
            e[j] = eigN(pp,args).ravel().sum()/(3*kp**2)
        de = np.gradient(e)
        dx = np.gradient(rangeP)
        der = de/dx
        f = interp1d(rangeP,der)
        res += (k[i]+f(P[i]))**2
    return res




