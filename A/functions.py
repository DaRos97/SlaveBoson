import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
from scipy.interpolate import interp1d
import inputs as inp
import time
from colorama import Fore,Style
#Some parameters from inputs.py
kp = inp.sum_pts
k = (inp.K1,inp.K2)
S = inp.S
J1 = inp.J1

#Function which returns e^(-i(k \dot a)) with k the 
#   momentum and a the position vector. m is the unit
#   cell size in real space, which can be 3 or 6.
#   a1 points to the right and a2 to up-left.
def exp_k(a1,a2):
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
def eig2G(P,args):#3x3 ansatx
    ti = time.time()
    A1 = P[0]
    Afn = P[1]
    L = P[2]
    J1,J2,J3,ans = args
    D = np.zeros((3,3,kp,kp),dtype=complex)
    D[0,0] += (1-ans)*J3*Afn*exp_k(1,0)
    D[0,1] += J1*A1*(exp_k(0,1)-1*(-1)**ans) - ans*J2*Afn*(exp_k(1,1)+exp_k(-1,0))
    D[0,2] += J1*A1*((-1)**ans*exp_k(-1,0)-exp_k(0,1)) + ans*J2*Afn*(1+exp_k(-1,1))
    D[1,1] += -(1-ans)*J3*Afn*exp_k(1,1)
    D[1,2] += J1*A1*(1-(-1)**ans*exp_k(-1,0)) - ans*J2*Afn*(exp_k(0,1)+exp_k(-1,-1))
    D[2,2] += (1-ans)*J3*Afn*exp_k(0,1)
    D[1,0] += -np.conjugate(D[0,1])
    D[2,0] += -np.conjugate(D[0,2])
    D[2,1] += -np.conjugate(D[1,2])
    res = np.zeros((3,kp,kp))
    for i in range(kp):
        for j in range(kp):
            D_ = np.conjugate(D[:,:,i,j]).T
            temp = LA.eigvalsh(np.matmul(D_,D[:,:,i,j]))
            res[0,i,j] = np.sqrt(L**2-temp[0]) if L**2-temp[0] > 0 else 0
            res[1,i,j] = np.sqrt(L**2-temp[1]) if L**2-temp[1] > 0 else 0
            res[2,i,j] = np.sqrt(L**2-temp[2]) if L**2-temp[2] > 0 else 0
    #print("Time eig2G: ",time.time()-ti)
    return res.ravel().sum()/(3*kp**2)

def tot_E(P,args):
    J1,J2,J3,ans = args
    zfn = inp.z[ans]
    Jfn = args[2-ans]
    E2 = 2*(inp.z1*J1*P[0]**2+zfn*Jfn*P[1]**2)-P[2]*(2*S+1)
    res = eig2G(P,args) + E2
    return res

def Sigma(P,args):
    ti = time.time()
    res = 0
    J1,J2,J3,ans = args
    Jfn = args[2-ans]
    zfn = inp.z[ans]
    ran = inp.der_range
    for i in range(len(P)):
        e = np.ndarray(inp.der_pts)
        rangeP = np.linspace(P[i]-ran[i],P[i]+ran[i],inp.der_pts)
        pp = np.array(P)
        for j in range(inp.der_pts):
            pp[i] = rangeP[j]
            e[j] = tot_E(pp,args)
        de = np.gradient(e)
        dx = np.gradient(rangeP)
        der = de/dx
        f = interp1d(rangeP,der)
        res += f(P[i])**2
    #print(Fore.BLUE+"Time Sigma: ",time.time()-ti,Style.RESET_ALL)
    return res




