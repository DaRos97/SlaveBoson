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

def sumE(A,L):
    m = 3
    N = np.zeros((m,m,grid_pts,grid_pts),dtype=complex)
    N[0,1] = J1*A*(exp_k(0,1)-1)
    N[0,2] = J1*A*(exp_k(-1,0)-exp_k(0,1))
    N[1,2] = J1*A*(1-exp_k(-1,0))
    res = np.zeros((m,grid_pts,grid_pts))
    for i in range(grid_pts):
        for j in range(grid_pts):
            N[:,:,i,j] -= np.conjugate(N[:,:,i,j]).T
            N_ = np.conjugate(N[:,:,i,j]).T
            temp = LA.eigvalsh(np.matmul(N[:,:,i,j],N_))
            for l in range(m):
                res[l,i,j] = np.sqrt(L**2-temp[l]) if L**2-temp[l] > 0 else 0
    func = (interp2d(kg[0],kg[1],res[0]),interp2d(kg[0],kg[1],res[1]),interp2d(kg[0],kg[1],res[2]))
    result = 0
    for i in range(m):
        temp = func[i](k3[0],k3[1])
        result += temp.ravel().sum()
    return result/(m*kp**2)
####
def minL(A):
    m = 3
    N = np.zeros((m,m,grid_pts,grid_pts),dtype=complex)
    N[0,1] = J1*A*(exp_k(0,1)-1)
    N[0,2] = J1*A*(exp_k(-1,0)-exp_k(0,1))
    N[1,2] = J1*A*(1-exp_k(-1,0))
    res = np.zeros((m,grid_pts,grid_pts))
    for i in range(grid_pts):
        for j in range(grid_pts):
            N[:,:,i,j] -= np.conjugate(N[:,:,i,j]).T
            N_ = np.conjugate(N[:,:,i,j]).T
            res[:,i,j] = LA.eigvalsh(np.matmul(N[:,:,i,j],N_))
    return np.sqrt(np.amax(res.ravel()))

####
def Sigma(A):
    #get L
    #L = minL(A) #since I know we are in a LRO phase
    #L = minimize_scalar(lambda l: derL(A,l),
    #            bounds = (mL,100),
    #            method = 'bounded'
    #            ).x
    #derivative wrt A
    der_pts = inp.der_pts
    der_ran = inp.der_range[0]
    E = np.ndarray(der_pts)
    ranA = np.linspace(A-der_ran,A+der_ran,der_pts)
    for j in range(der_pts):
        E[j] = totE(ranA[j])
    dE = np.gradient(E)
    da = np.gradient(ranA)
    der = dE/da
    func = interp1d(ranA,der)
    res = func(A)**2
    return res
####
def derL(A,L):
    der_pts = inp.der_pts
    der_ran = inp.der_range[2]
    E = np.ndarray(der_pts)
    ranL = np.linspace(L-der_ran,L+der_ran,der_pts)
    for j in range(der_pts):
        E[j] = totE(A,ranL[j])
    dE = np.gradient(E)
    dp = np.gradient(ranL)
    der = dE/dp
    func = interp1d(ranL,der)
    res = np.abs(func(L))
    return res
####
def totE(A):
    L = minL(A)
    eN = sumE(A,L)
    E2 = inp.z1*inp.J1*A**2 - L*(2*S+1) + inp.z1*J1*inp.S**2/2
    res = eN + E2
    return res
####
def getL(A):
    L = minimize_scalar(lambda l: derL(A,l),
                bounds = (0,10),
                method = 'bounded'
                ).x
    return L





