import numpy as np
import inputs as inp
import common_functions as cf
from time import time as t
from scipy.optimize import differential_evolution as d_e
from pandas import read_csv
import csv
import sys
####### inputs
J1 = inp.J1
J2, J3 = (0,0)
phi = inp.range_phi[int(sys.argv[1])]
#######
filename = inp.DataDir+'testDM_phi='+str("{:4.4f}".format(phi)).replace('.','')+'.npy'
ansatze = ['3x3','q0']
Ti = t()
Pinitial = {'3x3':(0.51,0.17), 'q0':(0.51,0.17)}
Bnds = ((0,1),(-0.5,0.5))
print("Using phi = ",phi)
res = [phi]
for ans in ansatze:
    print("Using ansatz: ",ans)
    inp.DM1 = phi
    Tti = t()
    Args = (J1,J2,J3,ans)
    result = d_e(lambda x: cf.Sigma(x,Args),
            x0 = Pinitial[ans],
            bounds = Bnds,
            popsize = 15,
            maxiter = inp.MaxIter,
#            disp = True,
            tol = inp.cutoff,
            atol = inp.cutoff,
            workers = 1     #parallelization --> see if necessary
            )
    Pf = tuple(result.x)
    S = result.fun
    E,L = cf.totE(Pf,Args)[:2]
    res.append(E)
    print("Time of phi point : ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################
RES = np.array(res)
np.save(filename, RES)
print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################
