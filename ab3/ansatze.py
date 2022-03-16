import inputs as inp
import numpy as np
from scipy import linalg as LA



grid_pts = inp.grid_pts
####
def exp_k(a1,a2):
    ax = a1+a2*(-1/2)
    ay = a2*np.sqrt(3)/2
    res = np.ndarray((grid_pts,grid_pts),dtype=complex)
    for i in range(grid_pts):
        for j in range(grid_pts):
            res[i,j] = np.exp(-1j*(inp.kg['m3'][0][i]*ax+inp.kg['m3'][1][j]*ay))
    return res
#### 3x3 ansatz
def sqrt3(P,L,args):
    m = 3
    J1,J2,J3,ans = args
    J1 /= 2.
    J2 /= 2.
    J3 /= 2.
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
    ##diagonal terms -> B3 and L
    N[0,0] = J3*B3*exp_k(1,0) + L
    N[1,1] = J3*B3*exp_k(1,1) + L
    N[2,2] = J3*B3*exp_k(0,1) + L
    N[3,3] = np.conjugate(N[0,0])     ###true?
    N[4,4] = np.conjugate(N[1,1])
    N[5,5] = np.conjugate(N[2,2])
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
    N[3,3] = np.conjugate(N[0,0])     ###true?
    N[4,4] = np.conjugate(N[1,1])
    N[5,5] = np.conjugate(N[2,2])
    #multiply by tau 3
    for i in range(m,2*m):
        for j in range(2*m):
            N[i,j] *= -1
    return N
#### 6 UC
def exp_k6(a1,a2):
    ax = a1-a2
    ay = a2*np.sqrt(3)
    res = np.ndarray((grid_pts,grid_pts),dtype=complex)
    for i in range(grid_pts):
        for j in range(grid_pts):
            res[i,j] = np.exp(-1j*(inp.kg['m6'][0][i]*ax+inp.kg['m6'][1][j]*ay))
    return res
#### (0,pi) ansatz
def zeroPi(P,L,args):
    m = 6
    J1,J2,J3,ans = args
    J1 /= 2
    J2 /= 2
    J3 /= 2
    A1,A2,A3,B1,B2 = P
    B1_ = -B1
    B2_ = -B2
    N = np.zeros((2*m,2*m,grid_pts,grid_pts),dtype=complex)
    #B1 and B2
    N[0,1] = J1*B1 + J2*B2*exp_k6(-1,0)
    N[0,2] = J1*B1*exp_k6(-1,0) + J2*B2
    N[0,4] = J1*B1*exp_k6(0,1) + J2*B2_*exp_k6(1,1)
    N[0,5] = J1*B1*exp_k6(0,1) + J2*B2_*exp_k6(-1,1)
    N[1,2] = J1*B1*(1+exp_k(-1,0))
    N[1,3] = J1*B1 + J2*B2*exp_k6(-1,0)
    N[1,5] = J2*(B2*exp_k6(0,1) + B2_*exp_k6(-1,0))
    N[2,3] = J1*B1 + J2*B2*exp_k6(1,0)
    N[2,4] = J2*B2*(1+exp_k6(1,1))
    N[3,4] = J1*B1 + J2*B2_*exp_k6(-1,0)
    N[3,5] = J1*B1_*exp_k6(-1,0) + J2*B2
    N[4,5] = J1*(B1 + B1_*exp_k6(-1,0))
    for i in range(m-1):
        for j in range(i+1,m):
            N[i+m,j+m] = N[i,j]
    #A1 and A2
    N[0,6] = J3*A3*exp_k6(1,0)
    N[0,7] = J1*A1 - J2*A2*exp_k6(-1,0)
    N[0,8] = -J1*A1*exp_k6(-1,0) + J2*A2
    N[0,10] = J1*A1*exp_k6(0,1) + J2*A2*exp_k6(1,1)
    N[0,11] = -J1*A1*exp_k6(0,1) - J2*A2*exp_k6(-1,1)
    N[1,8] = J1*A1*(1 + exp_k6(-1,0))
    N[1,9] = -J1*A1 + J2*A2*exp_k6(-1,0)
    N[1,10] = J3*A3*(exp_k6(1,1) + exp_k6(-1,0))
    N[1,11] = J2*A2*(exp_k6(-1,0) - exp_k6(0,1))
    N[2,9] = J1*A1 - J2*A2*exp_k6(1,0)
    N[2,10] = J2*A2*(1 + exp_k6(1,1))
    N[2,11] = J3*A3*(1 - exp_k6(0,1))
    N[3,9] = -J3*A3*exp_k6(1,0)
    N[3,10] = J1*A1 + J2*A2*exp_k6(1,0)
    N[3,11] = J1*A1*exp_k6(-1,0) + J2*A2
    N[4,11] = J1*A1*(1 - exp_k6(-1,0))
    for i in range(m-1):
        for j in range(m+i+1,2*m):
            N[j-m,i+m] = -np.conjugate(N[i,j])
    #complex conj
    for i in range(grid_pts):
        for j in range(grid_pts):
            N[:,:,i,j] += np.conjugate(N[:,:,i,j].T)
    ##diagonal terms
    for i in range(2*m):
        N[i,i] = L
    #multiply by tau 3
    for i in range(m,2*m):
        for j in range(2*m):
            N[i,j] *= -1
    return N
#### (pi,pi) ansatz
def PiPi(P,L,args):
    m = 6
    J1,J2,J3,ans = args
    J1 /= 2
    J2 /= 2
    J3 /= 2
    A1,B1,B2 = P
    B1_ = -B1
    B2_ = -B2
    N = np.zeros((2*m,2*m,grid_pts,grid_pts),dtype=complex)
    #B1 and B2
    N[0,1] = J1*B1 + J2*B2*exp_k6(-1,0)
    N[0,2] = J1*B1*exp_k6(-1,0) + J2*B2
    N[0,4] = J1*B1*exp_k6(0,1) + J2*B2_*exp_k6(1,1)
    N[0,5] = J1*B1*exp_k6(0,1) + J2*B2_*exp_k6(-1,1)
    N[1,2] = J1*B1*(1+exp_k(-1,0))
    N[1,3] = J1*B1 + J2*B2*exp_k6(-1,0)
    N[1,5] = J2*(B2*exp_k6(0,1) + B2_*exp_k6(-1,0))
    N[2,3] = J1*B1 + J2*B2*exp_k6(1,0)
    N[2,4] = J2*B2*(1+exp_k6(1,1))
    N[3,4] = J1*B1 + J2*B2_*exp_k6(-1,0)
    N[3,5] = J1*B1_*exp_k6(-1,0) + J2*B2
    N[4,5] = J1*(B1 + B1_*exp_k6(-1,0))
    for i in range(m-1):
        for j in range(i+1,m):
            N[i+m,j+m] = N[i,j]
    #A1 and A2
    N[0,7] = -J1*A1
    N[0,8] = J1*A1*exp_k6(-1,0)
    N[0,10] = J1*A1*exp_k6(0,1)
    N[0,11] = -J1*A1*exp_k6(0,1)
    N[1,8] = J1*A1*(1 - exp_k6(-1,0))
    N[1,9] = -J1*A1
    N[2,9] = J1*A1
    N[3,10] = -J1*A1
    N[3,11] = -J1*A1*exp_k6(-1,0)
    N[4,11] = J1*A1*(1 - exp_k6(-1,0))
    for i in range(m-1):
        for j in range(m+i+1,2*m):
            N[j-m,i+m] = -np.conjugate(N[i,j])
    #complex conj
    for i in range(grid_pts):
        for j in range(grid_pts):
            N[:,:,i,j] += np.conjugate(N[:,:,i,j].T)
    ##diagonal terms
    for i in range(2*m):
        N[i,i] = L
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
