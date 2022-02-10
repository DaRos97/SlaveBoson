import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import functions as fs
import inputs as inp
import time
from colorama import Style,Fore
start_time = time.time()
#dirname = 'Data3/'

#Fixed parameters
cutoff = inp.cutoff
z1 = inp.z1
J1 = inp.J1
S = inp.S
kp = inp.sum_pts
K1 = inp.K1
K2 = inp.K2

Stay = True
A = 0.2#2#6#37
B = 0.01#0#5#74
L = 0.2#4#1#82

Pa = np.zeros((3,inp.maxCicles))
Pa[0,0] = A
Pa[1,0] = B
Pa[2,0] = L

### cicle
cicle = 1
Energy_i = fs.tot_E(A,B,L)
print("Initial energy with parameters ",A,B,L," is ",Energy_i)
t = time.time()
sigma = fs.Sigma(A,B,L,Pa[:,cicle-1])
print("Initial sigma:",sigma)
print("time: ",time.time()-t)

T1 = time.time()
Stay = True
T1 = time.time()
while Stay:
    ti = time.time()
    print(Fore.RED,"initiating cicle ",cicle,Style.RESET_ALL)
    # A
    A = Pa[0,cicle-1]
    Aa = Pa[0,cicle-2]
    ran = 2*np.abs(A-Aa)
    if ran == 0:
        ran = 0.1
    mA = A-ran if A-ran > 0 else 0
    MA = A+ran if A+ran < (2*S+1)/2 else (2*S+1)/2
    minA = minimize_scalar(lambda a:fs.Sigma(a,B,L,Pa[:,cicle-1]), 
            method = 'bounded', bounds = (mA,MA))
    newA = minA.x
    print("old A: ",A)
    print("new A: ",newA)
    A = newA
    Pa[0,cicle] = A
    # B
    B = Pa[1,cicle-1]
    Ba = Pa[1,cicle-2]
    ran = 2*np.abs(B-Ba)
    if ran == 0:
        ran = 0.1
    mB = B-ran if B-ran > 0 else 0
    MB = B+ran if B+ran < S else S
    minB = minimize_scalar(lambda b:fs.Sigma(A,b,L,Pa[:,cicle-1]), 
            method = 'bounded', bounds = (mB,MB))
    newB = minB.x
    print("old B: ",B)
    print("new B: ",newB)
    B = newB
    Pa[1,cicle] = B
    # L
    L = Pa[2,cicle-1]
    La = Pa[2,cicle-2]
    ran = 2*np.abs(L-La)
    if ran == 0:
        ran = 0.1
    mL = L-ran if L-ran > 0 else 0
    ML = L+ran
    minL = minimize_scalar(lambda l:fs.Sigma(A,B,l,Pa[:,cicle-1]), 
            method = 'bounded', bounds = (mL,ML))
    newL = minL.x
    print("old L: ",L)
    print("new L: ",newL)
    L = newL
    Pa[2,cicle] = L
    #### Sigma
    sigma = fs.Sigma(A,B,L,Pa[:,cicle-1])
    print("New sigma:",sigma)
    if sigma < cutoff or cicle == inp.maxCicles:
        Stay = False
    else:
        print("cicle ",cicle," took ",time.time()-ti)
        cicle += 1

Energy_f = fs.tot_E(A,B,L)
print(Fore.GREEN,"Final energy with parameters ",A,B,L," is ",Energy_f)
print("Total time: ",time.time()-T1,Style.RESET_ALL)






