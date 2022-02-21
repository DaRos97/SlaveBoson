import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d,interp2d
import inputs as inp
import time
from colorama import Fore,Style
#Some parameters from inputs.py
kp = inp.sum_pts
k3 = (inp.K1,inp.K23)
k6 = (inp.K1,inp.K26)
S = inp.S
J1 = inp.J1
#grid points
grid_pts = inp.grid_pts
kg = (np.linspace(0,inp.maxK1,grid_pts),np.linspace(0,inp.maxK2,grid_pts))
####
def exp_k(a1,a2):
    ax = a1+a2*(-1/2)
    ay = a2*(np.sqrt(3)/2)
    res = np.ndarray((grid_pts,grid_pts),dtype=complex)
    for i in range(grid_pts):
        for j in range(grid_pts):
            res[i,j] = np.exp(-1j*(kg[0][i]*ax+kg[1][j]*ay))
    return res
####
def eigs(P,args):
    A1 = P[0]
    A2 = -P[1]
    A3 = -P[2]
    J1,J2,J3,ans = args
    m = 3
    D = np.zeros((m,m,grid_pts,grid_pts),dtype=complex)
    D[0,0] += (1-ans)*J3*A3*exp_k(1,0)
    D[0,1] += J1*A1*(exp_k(0,1)-1*(-1)**ans) - ans*J2*A2*(exp_k(1,1)+exp_k(-1,0))
    D[0,2] += J1*A1*((-1)**ans*exp_k(-1,0)-exp_k(0,1)) + ans*J2*A2*(1+exp_k(-1,1))
    D[1,1] += -(1-ans)*J3*A3*exp_k(1,1)
    D[1,2] += J1*A1*(1-(-1)**ans*exp_k(-1,0)) - ans*J2*A2*(exp_k(0,1)+exp_k(-1,-1))
    D[2,2] += (1-ans)*J3*A3*exp_k(0,1)
    D[1,0] -= np.conjugate(D[0,1])
    D[2,0] -= np.conjugate(D[0,2])
    D[2,1] -= np.conjugate(D[1,2])
    #grid of points
    res = np.zeros((m,grid_pts,grid_pts))
    for i in range(grid_pts):
        for j in range(grid_pts):
            D_ = np.conjugate(D[:,:,i,j]).T
            res[:,i,j] = LA.eigvalsh(np.matmul(D_,D[:,:,i,j]))
    return res
####
def sumEigs(P,L,args):
    temp = eigs(P,args)
    m = 3
    res = np.sqrt(L**2-temp)
    func = (interp2d(kg[0],kg[1],res[0]),interp2d(kg[0],kg[1],res[1]),interp2d(kg[0],kg[1],res[2]))
    result = 0
    for i in range(m):
        temp = func[i](k3[0],k3[1])
        result += temp.ravel().sum()
    return result/(m*kp**2)
####
def totE(P,args):
    L,mL = minL(P,args)
    res = totEl(P,L,args)
    return res, L, mL
####
def totEl(P,L,args):
    J1,J2,J3,ans = args
    J = (J1,J2,J3)
    res = 0
    for i in range(len(P)):
        res += inp.z[i]*P[i]**2*J[i] + inp.z[i]*J[i]*inp.S**2/2
    res -= L*(2*inp.S+1)
    res += sumEigs(P,L,args)
    return res
####
def Sigma(P,args):
    J1,J2,J3,ans = args
    J = (J1,J2,J3)
    tJ = ('J1','J2','J3')
    res = 0
    ran = inp.der_range
    for i in range(len(P)):
        if J[i] == 0:
            continue
        e = np.ndarray(inp.der_pts)
        rangeP = np.linspace(P[i]-ran[i],P[i]+ran[i],inp.der_pts)
        pp = np.array(P)
        for j in range(inp.der_pts):
            pp[i] = rangeP[j]
            e[j] = totE(pp,args)[0]        #uses at each energy evaluation the best lambda -> consuming
        de = np.gradient(e)
        dx = np.gradient(rangeP)
        der = de/dx
        f = interp1d(rangeP,der)
        res += f(P[i])**2
    return res
####
def minL(P,args):
    mL = np.sqrt(np.amax(eigs(P,args).ravel()))
    res = minimize_scalar(lambda l: -totEl(P,l,args),
            method = 'bounded',
            bounds = (mL,10),
            options={'xatol':1e-8}
            )
    L = res.x
    return L,mL

