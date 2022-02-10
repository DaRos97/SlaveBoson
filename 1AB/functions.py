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
def eigN(A,B,L):
    J1 = inp.J1
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
    return res

def tot_E(A,B,L):
    t = time.time()
    eN = eigN(A,B,L)
    print("eigN takes :",time.time()-t)
    E2 = 2*inp.J1*(A**2-B**2)-L*(2*S+1)
    res = eN.ravel().sum()/(3*kp*kp) + E2
    return res

########## DERIVATIVES
def sum_derA(A,B,L,Aa):
    J1 = inp.J1
    res1 = 4*J1*A
    pts = inp.derPts
    ran = 2*np.abs(A-Aa)
    if ran == 0:
        ran = 0.1
    Amin = A-ran if A-ran > 0 else 0
    Amax = A+ran if A+ran < (2*S+1)/2 else (2*S+1)/2
    range_A = np.linspace(Amin,Amax,pts)
    e1 = np.ndarray(pts)
    for i,Ai in enumerate(range_A):
        e1[i] = eigN(Ai,B,L).ravel().sum()/(3*kp*kp)
    de1 = np.gradient(e1)
    dx = np.gradient(range_A)
    der = de1/dx
    func = interp1d(range_A,der)
    res2 = func(A)
    res = (res1 + res2)**2
    return res

def sum_derB(A,B,L,Ba):
    J1 = inp.J1
    res1 = 4*J1*B
    pts = inp.derPts
    ran = 2*np.abs(B-Ba)
    if ran == 0:
        ran = 0.1
    Bmin = B-ran if B-ran > 0 else 0
    Bmax = B+ran if B+ran < S else S
    range_B = np.linspace(Bmin,Bmax,pts)
    e1 = np.ndarray(pts)
    for i,Bi in enumerate(range_B):
        e1[i] = eigN(A,Bi,L).ravel().sum()/(3*kp*kp)
    de1 = np.gradient(e1)
    dx = np.gradient(range_B)
    der = de1/dx
    func = interp1d(range_B,der)
    res2 = func(B)
    res = (res2 - res1)**2
    return res

def sum_derL(A,B,L,La):
    J1 = inp.J1
    res1 = 2*S+1
    pts = inp.derPts
    ran = 2*np.abs(L-La)
    if ran == 0:
        ran = 0.1
    Lmin = L-ran if L-ran > 0 else 0
    Lmax = L+ran 
    range_L = np.linspace(Lmin,Lmax,pts)
    e1 = np.ndarray(pts)
    for i,Li in enumerate(range_L):
        e1[i] = eigN(A,B,Li).ravel().sum()/(3*kp*kp)
    de1 = np.gradient(e1)
    dx = np.gradient(range_L)
    der = de1/dx
    func = interp1d(range_L,der)
    res2 = func(L)
    res = (res2 - res1)**2
    return res

def Sigma(A,B,L,Pa):
    return sum_derA(A,B,L,Pa[0]) + sum_derB(A,B,L,Pa[1]) + sum_derL(A,B,L,Pa[2])






