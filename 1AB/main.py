import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import functions as fs
import inputs as inp
from tqdm import tqdm
import time
from colorama import Style,Fore
start_time = time.time()
temp = start_time
#dirname = 'Data3/'

#Fixed parameters
cutoff = inp.cutoff
z1 = inp.z1
J1 = inp.J1
S = inp.S
kp = inp.sum_pts

Stay = True
A = 0.26#37
B = 0.05#74
L = 0.41#82

K1 = inp.K1
K2 = inp.K2
### cicle
cicle = 1
Energy_i = fs.sum_E(A,B,L)
print("Initial energy with parameters ",A,B,L," is ",Energy_i)
while Stay:
    print(Fore.RED + "Cycle ",cicle,Style.RESET_ALL)
    sigma = fs.Sigma(A,B,L)
    print("Initial sigma:",sigma)
    min1 = minimize_scalar(lambda a:fs.Sigma(a,B,L),
                            method = 'bounded',
                            bounds = (A-0.01,A+0.01))
    newA = min1.x
    print("old A: ",A)
    print("new A: ",newA)
    sigma2 = fs.Sigma(newA,B,L)
    print("new sigma:",sigma2)
    A = newA
    min2 = minimize_scalar(lambda b:fs.Sigma(A,b,L),
                            method = 'bounded',
                            bounds = (B-0.01,B+0.01))
    newB = min2.x
    print("old B: ",B)
    print("new B: ",newB)
    sigma3 = fs.Sigma(A,newB,L)
    print("new sigma:",sigma3)
    B = newB
    min3 = minimize_scalar(lambda l:fs.Sigma(A,B,l),
                            method = 'bounded',
                            bounds = (L-0.01,L+0.01))
    newL = min3.x
    print("old L: ",L)
    print("new L: ",newL)
    sigma4 = fs.Sigma(A,B,newL)
    print("new sigma:",sigma4)
    L = newL
    if sigma4 < cutoff:
        Stay = False
    else:
        cicle += 1
        print("cicle ",cicle-1," took ",time.time()-temp)
        temp = time.time()

Energy_f = fs.sum_E(A,B,L)
print("Final energy with parameters ",A,B,L," is ",Energy_f)

print("Total time: ",time.time()-start_time)
