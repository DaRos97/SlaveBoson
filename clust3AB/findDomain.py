import inputs as inp
import ansatze as an
import numpy as np
import common_functions as cf
from scipy import linalg as LA
from scipy.interpolate import interp2d
from scipy.optimize import minimize_scalar
import time

#Some parameters from inputs.py
m = inp.m
kp = inp.sum_pts
grid_pts = inp.grid_pts
####
J = np.zeros((2*m,2*m))
for i in range(m):
    J[i,i] = -1
    J[i+m,i+m] = 1
####
scan_pts = 10
def Domain(bounds,args):
    nParams = len(bounds)
    arr = []
    for i in range(nParams):
        temp = np.linspace(bounds[i][0],bounds[i][1],scan_pts)
        arr.append(temp)
    Domain = np.ndarray((scan_pts,scan_pts,2))
    for i,Pi in enumerate(arr[0]):
        for j,Pj in enumerate(arr[1]):
            P = (Pi,Pj)
            res = minimize_scalar(lambda l: -cf.totEl(tuple(P)+(l,),args),
                    method = inp.L_method,
                    bounds = inp.L_bounds,
                    options={'xatol':inp.prec_L}
                    )
            L = res.x
            N = an.Nk(P,L,args)
            b = 1
            for ki in range(grid_pts):
                for kj in range(grid_pts):
                    try:
                        a = LA.cholesky(N[:,:,ki,kj])
                    except LA.LinAlgError:
                        b = 0
                        break
            Domain[i,j] = (P,b)
    return Domain
