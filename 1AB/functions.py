import numpy as np
from scipy import linalg as LA
from scipy.integrate import dblquad
from scipy.interpolate import interp1d
import inputs as inp

#Function which returns e^(-i(k \dot a)) with k the 
#   momentum and a the position vector. m is the unit
#   cell size in real space, which can be 3 or 6.
#   a1 points to the right and a2 to up-left.
def exp_k(k,a1,a2):
    ax = a1+a2*(-1/2)
    ay = a2*(np.sqrt(3)/2)
    kp = inp.sum_pts
    res = np.ndarray((kp,kp),dtype=complex)
    for i in range(kp):
        for j in range(kp):
            res[i,j] = np.exp(-1j*(k[0][i]*ax+k[1][j]*ay))
    return res
#Evaluates the eigenvalues of (G^dag G) for the four different
#   ansatze and for arrays of momenta k1,k2 in the BZ. 
#   Params contains the couplings J1, J2, the DM phase phi 
#   and theansatz considered.
def eigN(A,B,L,k):
    J1 = inp.J1
    kp = inp.sum_pts
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
            res[:,i,j] = np.sort(temp.real)[3:]
            if np.amax(np.abs(temp.imag)) > 1e-5:
                print("imaginary component discarded")
    return res

def sum_E(A,B,L):
    k = (inp.K1,inp.K2)
    kp = inp.sum_pts
    eN = eigN(A,B,L,k)
    E2 = 2*inp.J1*(A**2-B**2)-L*(2*inp.S+1)
    res = eN.ravel().sum()/(3*kp*kp) + E2
    return res
########## DERIVATIVES
def sum_derA(A,B,L):
    J1 = inp.J1
    k = (inp.K1,inp.K2)
    kp = inp.sum_pts
    res1 = 4*J1*A
    pts = inp.derPts
    range_A = np.linspace(A-0.01,A+0.01,pts)
    e1 = np.ndarray((pts,3,kp,kp))
    print("hh")
    for i,Ai in enumerate(range_A):
        e1[i] = eigN(Ai,B,L,k)
    res2 = np.ndarray((3,kp,kp))
    for i in range(kp):
        for j in range(kp):
            for l in range(3):
                de1 = np.gradient(e1[:,l,i,j])
                dx = np.gradient(range_A)
                der = de1/dx
                func = interp1d(range_A,der)
                res2[l,i,j] = func(A)
    res = res1+res2.ravel().sum()/(3*kp*kp)
    return res**2

def sum_derB(A,B,L):
    J1 = inp.J1
    k = (inp.K1,inp.K2)
    kp = inp.sum_pts
    res1 = 4*J1*B
    pts = inp.derPts
    range_B = np.linspace(B-0.001,B+0.001,pts)
    e1 = np.ndarray((pts,3,kp,kp))
    for i,Bi in enumerate(range_B):
        e1[i] = eigN(A,Bi,L,k)
    res2 = np.ndarray((3,kp,kp))
    for i in range(kp):
        for j in range(kp):
            for l in range(3):
                de1 = np.gradient(e1[:,l,i,j])
                dx = np.gradient(range_B)
                der = de1/dx
                func = interp1d(range_B,der)
                res2[l,i,j] = func(B)
    res = -res1+res2.ravel().sum()/(3*kp*kp)
    return res**2

def sum_derL(A,B,L):
    J1 = inp.J1
    k = (inp.K1,inp.K2)
    S = inp.S
    kp = inp.sum_pts
    res1 = 2*S+1
    pts = inp.derPts
    range_L = np.linspace(L-0.01,L+0.01,pts)
    e1 = np.ndarray((pts,3,kp,kp))
    for i,Li in enumerate(range_L):
        e1[i] = eigN(A,B,Li,k)
    res2 = np.ndarray((3,kp,kp))
    for i in range(kp):
        for j in range(kp):
            for l in range(3):
                de1 = np.gradient(e1[:,l,i,j])
                dx = np.gradient(range_L)
                der = de1/dx
                func = interp1d(range_L,der)
                res2[l,i,j] = func(L)
    res = -res1+res2.ravel().sum()/(3*kp*kp)
    return res**2

def Sigma(A,B,L):
    return sum_derA(A,B,L) + sum_derB(A,B,L) + sum_derL(A,B,L)






