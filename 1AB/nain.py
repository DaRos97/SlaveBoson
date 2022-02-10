import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import functions as fs
import inputs as inp
import time
from colorama import Style,Fore
start_time = time.time()

#Fixed parameters
cutoff = inp.cutoff
z1 = inp.z1
J1 = inp.J1
S = inp.S
kp = inp.sum_pts
K1 = inp.K1
K2 = inp.K2

Stay = True
minA = 0
maxA = (2*S+1)/2
minB = 0
maxB = S
minL = 0
maxL = 1

Stay = True
while Stay:
    pts = 3
    Aa = np.linspace(minA,maxA,pts)
    Ba = np.linspace(minB,maxB,pts)
    La = np.linspace(minL,maxL,pts)
    Ea = np.ndarray((pts,pts,pts))
    for a in Aa:
        for b in Ba:
            for l in La:
                Ea[a,b,l] = fs.Sigma(a,b,l)
    minE = 
    exit()


