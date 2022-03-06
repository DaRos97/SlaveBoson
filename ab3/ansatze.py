import inputs as inp
import numpy as np
from scipy import linalg as LA



grid_pts = inp.grid_pts
kg = (np.linspace(0,inp.maxK1,grid_pts),np.linspace(0,inp.maxK2,grid_pts))

####
def exp_k(a1,a2):
    ax = a1+a2*(-1/2)
    ay = a2*np.sqrt(3)/2
    res = np.ndarray((grid_pts,grid_pts),dtype=complex)
    for i in range(grid_pts):
        for j in range(grid_pts):
            res[i,j] = np.exp(-1j*(kg[0][i]*ax+kg[1][j]*ay))
    return res
#### 3x3 ansatz
def sqrt3(P,L,args):
    m = 3
    J1,J2,J3,ans = args
    J1 /= 2.
    J2 /= 2.
    J3 /= 2.
    A1,A3,B1,B2,B3 = P
    A3 *= -1.
    B1 *= -1.
    N = np.zeros((2*m,2*m,grid_pts,grid_pts),dtype=complex)
    #B1 and B2
    N[0,1] = J1*B1*(1+exp_k(0,1)) + J2*B2*(exp_k(1,1)+exp_k(-1,0))
    N[0,2] = J1*B1*(exp_k(0,1)+exp_k(-1,0)) + J2*B2*(1+exp_k(-1,1))
    N[1,2] = J1*B1*(1+exp_k(-1,0)) + J2*B2*(exp_k(0,1)+exp_k(-1,-1))
    N[3,4] = N[0,1]
    N[3,4] = N[0,2]
    N[4,5] = N[1,2]
    #A3
    N[0,3] = J3*A3*exp_k(1,0)
    N[1,4] = -J3*A3*exp_k(1,1)
    N[2,5] = J3*A3*exp_k(0,1)
    #A1 and A2
    N[0,4] = J1*A1*(exp_k(0,1)-1)# - J2*A2*(exp_k(1,1)+exp_k(-1,0))
    N[0,5] = J1*A1*(exp_k(-1,0)-exp_k(0,1))# + J2*A2*(1+exp_k(-1,1))
    N[1,5] = J1*A1*(1-exp_k(-1,0))# - J2*A2*(exp_k(0,1)+exp_k(-1,-1))
    N[1,3] = -np.conjugate(N[0,4])
    N[2,3] = -np.conjugate(N[0,5])
    N[2,4] = -np.conjugate(N[1,5])
    #complex conj
    for i in range(grid_pts):
        for j in range(grid_pts):
            N[:,:,i,j] += np.conjugate(N[:,:,i,j].T)
    ##diagonal terms
    N[0,0] = J3*B3*exp_k(1,0) + L
    N[1,1] = J3*B3*exp_k(1,1) + L
    N[2,2] = J3*B3*exp_k(0,1) + L
    N[3,3] = N[0,0]     ###true?
    N[4,4] = N[1,1]
    N[5,5] = N[2,2]
    #multiply by tau 3
    for i in range(m,2*m):
        for j in range(2*m):
            N[i,j] *= -1
    return N
#### Q=0 ansatz
def q0(P,L,args):
    m = 3
    J1,J2,J3,ans = args
    J1 /= 2
    J2 /= 2
    J3 /= 2
    A1,A2,B1,B2,B3 = P
    A2 *= -1.
    B1 *= -1.
    B2 *= -1.
    N = np.zeros((2*m,2*m,grid_pts,grid_pts),dtype=complex)
    #B1 and B2
    N[0,1] = J1*B1*(1+exp_k(0,1)) + J2*B2*(exp_k(1,1)+exp_k(-1,0))
    N[0,2] = J1*B1*(exp_k(0,1)+exp_k(-1,0)) + J2*B2*(1+exp_k(-1,1))
    N[1,2] = J1*B1*(1+exp_k(-1,0)) + J2*B2*(exp_k(0,1)+exp_k(-1,-1))
    N[3,4] = N[0,1]
    N[3,4] = N[0,2]
    N[4,5] = N[1,2]
    #A1 and A2
    N[0,4] += J1*A1*(exp_k(0,1)+1) - J2*A2*(exp_k(1,1)+exp_k(-1,0))
    N[0,5] += -J1*A1*(exp_k(-1,0)+exp_k(0,1)) + J2*A2*(1+exp_k(-1,1))
    N[1,5] += J1*A1*(1+exp_k(-1,0)) - J2*A2*(exp_k(0,1)+exp_k(-1,-1))
    N[1,3] -= np.conjugate(N[0,4])
    N[2,3] -= np.conjugate(N[0,5])
    N[2,4] -= np.conjugate(N[1,5])
    #complex conj
    for i in range(grid_pts):
        for j in range(grid_pts):
            N[:,:,i,j] += np.conjugate(N[:,:,i,j].T)
    ##diagonal terms
    N[0,0] = J3*B3*exp_k(1,0) + L
    N[1,1] = J3*B3*exp_k(1,1) + L
    N[2,2] = J3*B3*exp_k(0,1) + L
    N[3,3] = N[0,0]     ###true?
    N[4,4] = N[1,1]
    N[5,5] = N[2,2]
    #multiply by tau 3
    for i in range(m,2*m):
        for j in range(2*m):
            N[i,j] *= -1
    return N
#### Cuboc 1 ansatz
def cuboc1(P,L,args):
    m = 3
    J1,J2,J3,ans = args
    J1 /= 2
    J2 /= 2
    J3 /= 2
    A1,A3,B1,B2,B3 = P
    N = np.zeros((2*m,2*m,grid_pts,grid_pts),dtype=complex)
    #B1 and B2
    N[0,1] = J1*B1*(1+exp_k(0,1)) + J2*B2*(exp_k(1,1)+exp_k(-1,0))
    N[0,2] = J1*B1*(exp_k(0,1)+exp_k(-1,0)) + J2*B2*(1+exp_k(-1,1))
    N[1,2] = J1*B1*(1+exp_k(-1,0)) + J2*B2*(exp_k(0,1)+exp_k(-1,-1))
    N[3,4] = N[0,1]
    N[3,4] = N[0,2]
    N[4,5] = N[1,2]
    #A3
    N[0,3] += J3*A3*exp_k(1,0)
    N[1,4] += -J3*A3*exp_k(1,1)
    N[2,5] += J3*A3*exp_k(0,1)
    #A1 and A2
    N[0,4] += J1*A1*(exp_k(0,1)-1)# - J2*A2*(exp_k(1,1)+exp_k(-1,0))
    N[0,5] += J1*A1*(exp_k(-1,0)-exp_k(0,1))# + J2*A2*(1+exp_k(-1,1))
    N[1,5] += J1*A1*(1-exp_k(-1,0))# - J2*A2*(exp_k(0,1)+exp_k(-1,-1))
    N[1,3] -= np.conjugate(N[0,4])
    N[2,3] -= np.conjugate(N[0,5])
    N[2,4] -= np.conjugate(N[1,5])
    #complex conj
    for i in range(grid_pts):
        for j in range(grid_pts):
            N[:,:,i,j] += np.conjugate(N[:,:,i,j].T)
    ##diagonal terms
    N[0,0] = J3*B3*exp_k(1,0) + L
    N[1,1] = J3*B3*exp_k(1,1) + L
    N[2,2] = J3*B3*exp_k(0,1) + L
    N[3,3] = N[0,0]     ###true?
    N[4,4] = N[1,1]
    N[5,5] = N[2,2]
    #multiply by tau 3
    for i in range(m,2*m):
        for j in range(2*m):
            N[i,j] *= -1
    return N
