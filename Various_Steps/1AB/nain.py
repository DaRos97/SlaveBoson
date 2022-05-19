import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from scipy.interpolate import interp1d
import functions as fs
import inputs as inp
import time
from colorama import Style,Fore
start_time = time.time()

S = inp.S

Ai = 0.2##26
Bi = 0.01#05
Li = 0.2#41

minA = 0
maxA = (2*S+1)/2
minB = 0
maxB = S
minL = 0
maxL = 1

bnds = ((minA,maxA),(minB,maxB),(minL,maxL))

min1 = minimize(lambda x:fs.Sigma(x[0],x[1],x[2]), 
            (Ai,Bi,Li), 
            method = 'Powell',
            bounds = bnds,
            options={'xtol': 1e-4}
            )

print("Results:")
A,B,L = min1.x
s = fs.Sigma(A,B,L)
print("Min values found:",A,B,L)
print("Sigma : ",s)
e = fs.tot_E(A,B,L)
print(e)

print("Tima taken: ",time.time()-start_time)
