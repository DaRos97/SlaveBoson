import numpy as np
import inputs as inp
import common_functions as cf
from time import time as t
from scipy.optimize import differential_evolution as d_e
import sys
####### inputs
N = 0
J2, J3 = inp.J[N]
#######
csvfile = inp.DataDir+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
ansatze = ['3x3_1']
Ti = t()
Pinitial, done  = cf.FindInitialPoint(J2,J3,ansatze)
Bnds = cf.FindBounds(J2,J3,ansatze,done,Pinitial)
DerRange = cf.ComputeDerRanges(J2,J3,ansatze)
for ans in ansatze:
    header = inp.header[ans]
    Args = (inp.J1,J2,J3,ans,DerRange[ans])
    #
    Ti = t()
    result = d_e(cf.Sigma,
        args = Args,
        x0 = Pinitial[ans],
        bounds = Bnds[ans],
        popsize = 21,#inp.mp_cpu*2,
        maxiter = inp.MaxIter*len(Pinitial[ans]),
#        disp = True,
        tol = inp.cutoff,
        atol = inp.cutoff,
        updating='deferred' if inp.mp_cpu != 1 else 'immediate',
        workers = inp.mp_cpu     #parallelization
        )
    Pf = tuple(result.x)
    print("Minimization took: ",t()-Ti)
    Ti = t()
    try:
        S, HessVals, E, L, gap = cf.Final_Result(Pf,*Args)
    except TypeError:
        print("Not saving, an Hessian sign is not right")
        print("Found values: Pf=",Pf,"\nSigma = ",result.fun)
        print("Time of ans",ans,": ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################
        continue
    conv = cf.IsConverged(Pf,Bnds[ans],S)
    newP = cf.FormatParams(Pf,ans,J2,J3)
    data = [ans,J2,J3,conv,E,S,gap,L]
    DataDic = {}
    for ind in range(len(data)):
        DataDic[header[ind]] = data[ind]
    for ind2 in range(len(newP)):
        DataDic[header[len(data)+ind2]] = newP[ind2]
    #save values
    cf.SaveToCsv(DataDic,csvfile)
    print("Rest of loop took: ",t()-Ti)
