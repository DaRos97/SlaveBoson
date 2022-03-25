import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d,interp2d
import inputs as inp
import time
from colorama import Fore,Style

S = inp.S
J1 = inp.J1

kp = inp.sum_pts
grid_pts = inp.grid_pts
k3 = (inp.K1,inp.K23)
kg = (np.linspace(0,inp.maxK1,grid_pts),np.linspace(0,inp.maxK2,grid_pts))

def exp_k(a1,a2):
    ax = a1+a2*(-1/2)
    ay = a2*(np.sqrt(3)/2)
    res = np.ndarray((grid_pts,grid_pts),dtype=complex)
    for i in range(grid_pts):
        for j in range(grid_pts):
            res[i,j] = np.exp(-1j*(kg[0][i]*ax+kg[1][j]*ay))
    return res

def sumE(P,L):
    A = P[0]
    B = P[1]
    m = 3
    N = np.zeros((2*m,2*m,grid_pts,grid_pts),dtype=complex)
    N[0,1] = -J1/2*B*(1+exp_k(0,1))
    N[0,2] = -J1/2*B*(exp_k(0,1)+exp_k(-1,0))
    N[1,2] = -J1/2*B*(1+exp_k(-1,0))
    N[3,4] = N[0,1]
    N[3,5] = N[0,2]
    N[4,5] = N[1,2]
    N[0,4] = J1/2*A*(-exp_k(0,1)+1)
    N[0,5] = J1/2*A*(-exp_k(-1,0)+exp_k(0,1))
    N[1,5] = J1/2*A*(-1+exp_k(-1,0))
    N[1,3] = -np.conjugate(N[0,4])
    N[2,3] = -np.conjugate(N[0,5])
    N[2,4] = -np.conjugate(N[1,5])
    res = np.zeros((m,grid_pts,grid_pts))
    for i in range(grid_pts):
        for j in range(grid_pts):
            N[:,:,i,j] = N[:,:,i,j] + np.conjugate(N[:,:,i,j]).T    #c.c
            for l in range(2*m):    #diag lambda
                N[l,l,i,j] = L
            for I in range(m,2*m):    #*sigma_3
                for J in range(2*m):
                    N[I,J,i,j] *= -1
            temp = LA.eigvals(N[:,:,i,j])   #eigvals since it is not hermitian
            res[:,i,j] = np.sort(temp.real)[3:]     #problem of imaginary part -> is not 0
    func = (interp2d(kg[0],kg[1],res[0]),interp2d(kg[0],kg[1],res[1]),interp2d(kg[0],kg[1],res[2]))
    result = 0
    for i in range(m):
        temp = func[i](k3[0],k3[1])
        result += temp.ravel().sum()
    return result/(m*kp**2)
####
def Sigma(P):
    res = 0
    #get L
    L = minimize_scalar(lambda l: derL(P,l),#-totE(P,l),
                bounds = (0,1),
                method = 'bounded'
                ).x
    #derivatives wrt A and B
    der_pts = inp.der_pts
    der_ran = inp.der_range
    E = np.ndarray(der_pts)
    for i in range(len(P)):
        ranP = np.linspace(P[i]-der_ran[i],P[i]+der_ran[i],der_pts)
        Pcopy = np.array(P)
        for j in range(der_pts):
            Pcopy[i] = ranP[j]
            E[j] = totE(Pcopy,L)
        dE = np.gradient(E)
        dp = np.gradient(ranP)
        der = dE/dp
        func = interp1d(ranP,der)
        res += (func(P[i]))**2
    return res
####
def derL(P,L):
    der_pts = inp.der_pts
    der_ran = inp.der_range[2]
    E = np.ndarray(der_pts)
    ranL = np.linspace(L-der_ran,L+der_ran,der_pts)
    for j in range(der_pts):
        E[j] = totE(P,ranL[j])
    dE = np.gradient(E)
    dp = np.gradient(ranL)
    der = dE/dp
    func = interp1d(ranL,der)
    res = np.abs(func(L))
    return res
####
def totE(P,L):
    A = P[0]
    B = P[1]
    eN = sumE(P,L)
    E2 = 2*inp.J1*(A**2-B**2)-L*(2*S+1)
    res = eN + E2
    return res
####
def getL(P):
    L = minimize_scalar(lambda l: derL(P,l),#-totE(P,l),
                bounds = (0,1),
                method = 'bounded'
                ).x
    return L





