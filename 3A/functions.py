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
def exp_kg(a1,a2):
    ax = a1+a2*(-1/2)
    ay = a2*(np.sqrt(3)/2)
    res = np.ndarray((grid_pts,grid_pts),dtype=complex)
    for i in range(grid_pts):
        for j in range(grid_pts):
            res[i,j] = np.exp(-1j*(kg[0][i]*ax+kg[1][j]*ay))
    return res

def En3(P,args):#3x3 ansatx
    A1 = P[0]
    Afn = P[1]
    L = P[2]
    J1,J2,J3,ans = args
    m = 3
    D = np.zeros((m,m,grid_pts,grid_pts),dtype=complex)
    D[0,0] += (1-ans)*J3*Afn*exp_kg(1,0)
    D[0,1] += J1*A1*(exp_kg(0,1)-1*(-1)**ans) - ans*J2*Afn*(exp_kg(1,1)+exp_kg(-1,0))
    D[0,2] += J1*A1*((-1)**ans*exp_kg(-1,0)-exp_kg(0,1)) + ans*J2*Afn*(1+exp_kg(-1,1))
    D[1,1] += -(1-ans)*J3*Afn*exp_kg(1,1)
    D[1,2] += J1*A1*(1-(-1)**ans*exp_kg(-1,0)) - ans*J2*Afn*(exp_kg(0,1)+exp_kg(-1,-1))
    D[2,2] += (1-ans)*J3*Afn*exp_kg(0,1)
    D[1,0] += -np.conjugate(D[0,1])
    D[2,0] += -np.conjugate(D[0,2])
    D[2,1] += -np.conjugate(D[1,2])
    #grid of points
    res = np.zeros((m,grid_pts,grid_pts))
    for i in range(grid_pts):
        for j in range(grid_pts):
            D_ = np.conjugate(D[:,:,i,j]).T
            temp = LA.eigvalsh(np.matmul(D_,D[:,:,i,j]))
            for l in range(m):
                res[l,i,j] = np.sqrt(L**2-temp[l]) if L**2-temp[l] > 0 else 0
    func = (interp2d(kg[0],kg[1],res[0]),interp2d(kg[0],kg[1],res[1]),interp2d(kg[0],kg[1],res[2]))
    result = 0
    for i in range(m):
        temp = func[i](k3[0],k3[1])
        result += temp.ravel().sum()
    return 2*result/(m*kp**2)

def Tot_E3(P,args):
    J1,J2,J3,ans = args
    zfn = inp.z[ans]
    Jfn = args[2-ans]
    E2 = 2*(inp.z1*J1*P[0]**2+zfn*Jfn*P[1]**2)-P[2]*(2*S+1)
    res = En3(P,args) + E2
    return res

def Sigma3(P,args):
    res = 0
    J1,J2,J3,ans = args
    Jfn = args[2-ans]
    zfn = inp.z[ans]
    ran = inp.der_range3
    for i in range(len(P)):
        e = np.ndarray(inp.der_pts)
        rangeP = np.linspace(P[i]-ran[i],P[i]+ran[i],inp.der_pts)
        pp = np.array(P)
        for j in range(inp.der_pts):
            pp[i] = rangeP[j]
            e[j] = Tot_E3(pp,args)
        de = np.gradient(e)
        dx = np.gradient(rangeP)
        der = de/dx
        f = interp1d(rangeP,der)
        res += f(P[i])**2
    return res
############################################################### good minimization
def En3L(P,args):#3x3 ansatx
    A1 = P[0]
    Afn = P[1]
    J1,J2,J3,ans,L = args
    m = 3
    D = np.zeros((m,m,grid_pts,grid_pts),dtype=complex)
    D[0,0] += (1-ans)*J3*Afn*exp_kg(1,0)
    D[0,1] += J1*A1*(exp_kg(0,1)-1*(-1)**ans) - ans*J2*Afn*(exp_kg(1,1)+exp_kg(-1,0))
    D[0,2] += J1*A1*((-1)**ans*exp_kg(-1,0)-exp_kg(0,1)) + ans*J2*Afn*(1+exp_kg(-1,1))
    D[1,1] += -(1-ans)*J3*Afn*exp_kg(1,1)
    D[1,2] += J1*A1*(1-(-1)**ans*exp_kg(-1,0)) - ans*J2*Afn*(exp_kg(0,1)+exp_kg(-1,-1))
    D[2,2] += (1-ans)*J3*Afn*exp_kg(0,1)
    D[1,0] += -np.conjugate(D[0,1])
    D[2,0] += -np.conjugate(D[0,2])
    D[2,1] += -np.conjugate(D[1,2])
    #grid of points
    res = np.zeros((m,grid_pts,grid_pts))
    for i in range(grid_pts):
        for j in range(grid_pts):
            D_ = np.conjugate(D[:,:,i,j]).T
            temp = LA.eigvalsh(np.matmul(D_,D[:,:,i,j]))
            for l in range(m):
                res[l,i,j] = np.sqrt(L**2-temp[l]) if L**2-temp[l] > 0 else 0
    func = (interp2d(kg[0],kg[1],res[0]),interp2d(kg[0],kg[1],res[1]),interp2d(kg[0],kg[1],res[2]))
    result = 0
    for i in range(m):
        temp = func[i](k3[0],k3[1])
        result += temp.ravel().sum()
    return 2*result/(m*kp**2)
def Tot_E3L(P,args):
    J1,J2,J3,ans,L = args
    zfn = inp.z[ans]
    Jfn = args[2-ans]
    L = maxE(P,args)
    E2 = 2*(inp.z1*J1*P[0]**2+zfn*Jfn*P[1]**2)-L*(2*S+1)
    res = En3L(P,args) + E2
    return res,L
def E3L(P,args):
    J1,J2,J3,ans,L = args
    zfn = inp.z[ans]
    Jfn = args[2-ans]
    E2 = 2*(inp.z1*J1*P[0]**2+zfn*Jfn*P[1]**2)-L*(2*S+1)
    res = En3L(P,args) + E2
    return res
def maxE(P,args):
    J1,J2,J3,ans,L = args
    #maximization wrt L
    minL = minimize_scalar(lambda l:E3L(P,(J1,J2,J3,ans,l)),
                bounds = (0,1),
                method = 'bounded'
                ).x
    L = minL
    return L
def SigmaA(P,args):
    res = 0
    J1,J2,J3,ans,L = args
    #maximization wrt L
    minL = minimize_scalar(lambda l:-E3L(P,(J1,J2,J3,ans,l)),
                bounds = (0,1),
                method = 'bounded'
                ).x
    L = minL
    args = (J1,J2,J3,ans,L)
    Jfn = args[2-ans]
    zfn = inp.z[ans]
    ran = inp.der_range3
    for i in range(len(P)):
        e = np.ndarray(inp.der_pts)
        rangeP = np.linspace(P[i]-ran[i],P[i]+ran[i],inp.der_pts)
        pp = np.array(P)
        for j in range(inp.der_pts):
            pp[i] = rangeP[j]
            e[j] = Tot_E3L(pp,args)[0]
        de = np.gradient(e)
        dx = np.gradient(rangeP)
        der = de/dx
        f = interp1d(rangeP,der)
        res += f(P[i])**2
    return res,minL
####################################### AB
def Eab(P):
    A = P[0]
    B = P[1]
    L = P[2]
    N = np.zeros((6,6,grid_pts,grid_pts),dtype=complex)
    N[0,1] = -J1/2*B*(1+exp_kg(0,1))
    N[0,2] = -J1/2*B*(exp_kg(0,1)+exp_kg(-1,0))
    N[1,2] = -J1/2*B*(1+exp_kg(-1,0))
    N[3,4] = N[0,1]
    N[3,5] = N[0,2]
    N[4,5] = N[1,2]
    N[0,4] = J1/2*A*(exp_kg(0,1)-1)
    N[0,5] = J1/2*A*(exp_kg(-1,0)-exp_kg(0,1))
    N[1,5] = J1/2*A*(1-exp_kg(-1,0))
    N[1,3] = -np.conjugate(N[0,4])
    N[2,3] = -np.conjugate(N[0,5])
    N[2,4] = -np.conjugate(N[1,5])
    res = np.zeros((3,grid_pts,grid_pts))
    for i in range(grid_pts):
        for j in range(grid_pts):
            N[:,:,i,j] = N[:,:,i,j] + np.conjugate(N[:,:,i,j]).T
            for l in range(6):
                N[l,l,i,j] = L
            for m in range(3,6):
                for n in range(6):
                    N[m,n,i,j] *= -1
            temp = LA.eigvals(N[:,:,i,j])
            res[:,i,j] = np.sort(temp.real)[3:]             #problem of imaginary part -> is not 0
    func = (interp2d(kg[0],kg[1],res[0]),interp2d(kg[0],kg[1],res[1]),interp2d(kg[0],kg[1],res[2]))
    result = 0
    for i in range(3):
        temp = func[i](k3[0],k3[1])
        result += temp.ravel().sum()
    return result/(3*kp**2)

def Tot_Eab(P):
    A,B,L = P
    L = maxL(P)
    eN = Eab((A,B,L))
    E2 = 2*inp.J1*(A**2-B**2)-L*(2*S+1)
    res = eN + E2
    return res,L
def maxL(P):
    #maximization wrt L
    minL = minimize_scalar(lambda l:-E3Lab((P[0],P[1],l)),
                bounds = (0,1),
                method = 'bounded'
                ).x
    L = minL
    return L
def E3Lab(P):
    A,B,L = P
    eN = Eab(P)
    E2 = 2*inp.J1*(A**2-B**2)-L*(2*S+1)
    res = eN + E2
    return res
def SigmaLab(P):
    pts = inp.der_pts
    res = 0
    p = [P[0],P[1],P[2]]
    k = [4*p[0]*J1,-4*p[1]*J1,-2*S-1]
    ran = inp.der_range3
    e = np.ndarray(pts)
    for i in range(len(p)):
        rangeP = np.linspace(p[i]-ran[i],p[i]+ran[i],pts)
        pp = p
        for j in range(pts):
            pp[i] = rangeP[j]
            e[j] = Eab(pp)
        de = np.gradient(e)
        dx = np.gradient(rangeP)
        der = de/dx
        f = interp1d(rangeP,der)
        res += (k[i]+f(p[i]))**2
    return res
    res = 0
    #maximization wrt L
    minL = minimize_scalar(lambda l:-E3L(P,(J1,J2,J3,ans,l)),
                bounds = (0,1),
                method = 'bounded'
                ).x
    L = minL
    args = (J1,J2,J3,ans,L)
    Jfn = args[2-ans]
    zfn = inp.z[ans]
    ran = inp.der_range3
    for i in range(len(P)):
        e = np.ndarray(inp.der_pts)
        rangeP = np.linspace(P[i]-ran[i],P[i]+ran[i],inp.der_pts)
        pp = np.array(P)
        for j in range(inp.der_pts):
            pp[i] = rangeP[j]
            e[j] = Tot_E3L(pp,args)[0]
        de = np.gradient(e)
        dx = np.gradient(rangeP)
        der = de/dx
        f = interp1d(rangeP,der)
        res += f(P[i])**2
    return res,minL



############################################################################################
def exp_k6(a1,a2):
    ax = a1+a2*(-1/2)*2
    ay = a2*(np.sqrt(3)/2)*2
    res = np.ndarray((kp,kp),dtype=complex)
    for i in range(kp):
        for j in range(kp):
            res[i,j] = np.exp(-1j*(k6[0][i]*ax+k6[1][j]*ay))
    return res

def en6(P,args):
    A1 = P[0]
    A2 = P[1]
    A3 = P[2]
    L = P[3]
    m = 6
    J1,J2,J3,ans = args
    D = np.zeros((m,m,kp,kp),dtype=complex)
    D[0,1] += J1*A1 - J2*A2*exp_k6(-1,0)
    D[0,2] += -J1*A1*exp_k6(-1,0) + J2*A2
    D[0,4] += J1*A1*exp_k6(0,1) + J2*A2*exp_k6(1,1)
    D[0,5] += -J1*A1*exp_k6(0,1) - J2*A2*exp_k6(-1,1)
    D[1,2] += J1*A1*(1+exp_k6(-1,0))
    D[1,3] += -J1*A1+J2*A2*exp_k6(-1,0)
    D[1,4] += J3*A3*(exp_k6(1,1)+exp_k6(-1,0))
    D[1,5] += J2*A2*(exp_k6(-1,0)-exp_k6(0,1))
    D[2,3] += J1*A1-J2*A2*exp_k6(1,0)
    D[2,4] += J2*A2*(1+exp_k6(1,1))
    D[2,5] += J3*A3*(1-exp_k6(0,1))
    D[3,4] += J1*A1+J2*A2*exp_k6(1,0)
    D[3,5] += J1*A1*exp_k6(-1,0)+J2*A2
    D[4,5] += J1*A1*(1-exp_k6(-1,0))
    #c.c
    for i in range(m):
        for j in range(m):
            D[i,j] -= np.conjugate(D[j,i])
    #diag elements
    D[0,0] += J3*A3*exp_k6(1,0)
    D[3,3] += -J3*A3*exp_k6(1,0)
    res = np.zeros((m,kp,kp))
    for i in range(kp):
        for j in range(kp):
            D_ = np.conjugate(D[:,:,i,j]).T
            temp = LA.eigvalsh(np.matmul(D_,D[:,:,i,j]))
            for l in range(m):
                res[l,i,j] = np.sqrt(L**2-temp[l]) if L**2-temp[l] > 0 else 0
    return res.ravel().sum()/(m*kp**2)

def tot_E6(P,args):
    J1,J2,J3,ans = args
    E2 = 2*(inp.z1*J1*P[0]**2+inp.z2*J2*P[1]**2+inp.z3*J3*P[2]**2)-P[3]*(2*S+1)
    res = en6(P,args) + E2
    return res

def Sigma6(P,args):
    res = 0
    J1,J2,J3,ans = args
    ran = inp.der_range6
    for i in range(len(P)): #4
        e = np.ndarray(inp.der_pts)
        rangeP = np.linspace(P[i]-ran[i],P[i]+ran[i],inp.der_pts)
        pp = np.array(P)
        for j in range(inp.der_pts):
            pp[i] = rangeP[j]
            e[j] = tot_E6(pp,args)
        de = np.gradient(e)
        dx = np.gradient(rangeP)
        der = de/dx
        f = interp1d(rangeP,der)
        res += f(P[i])**2
    return res


