import numpy as np
import inputs as inp
import common_functions as cf
from time import time as t
from scipy.optimize import differential_evolution as d_e
import sys
####### inputs
N = int(sys.argv[1])
J2, J3 = inp.J[N]
print('\n(J2,J3) = ('+'{:5.4f}'.format(J2)+',{:5.4f}'.format(J3)+')\n')
#######
csvfile = inp.DataDir+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
ansatze = cf.CheckCsv(csvfile)
Ti = t()
Pinitial, done  = cf.FindInitialPoint(J2,J3,ansatze)
Bnds = cf.FindBounds(J2,J3,ansatze,done,Pinitial)
DerRange = cf.ComputeDerRanges(J2,J3,ansatze)
for ans in ansatze:
    Tti = t()
    header = inp.header[ans]
    #
    j2 = int(np.sign(J2)*np.sign(int(np.abs(J2)*1e8)) + 1)   #j < 0 --> 0, j == 0 --> 1, j > 0 --> 2
    j3 = int(np.sign(J3)*np.sign(int(np.abs(J3)*1e8)) + 1)
    pars2 = inp.Pi[ans].keys()
    pars = []
    for pPp in pars2:
        if (pPp[-1] == '1') or (pPp[-1] == '2' and j2-1) or (pPp[-1] == '3' and j3-1):
            pars.append(pPp)
    hess_sign = {}
    for par in pars:
        if par[-2] == 'A':
            if par[-1] == '1' or (par[-1] == '2' and J2 > 0) or (par[-1] == '3' and J3 > 0):
                hess_sign[par] = 1
            else:
                hess_sign[par] = -1
        else:
            if par[-1] == '1' or (par[-1] == '2' and J2 > 0) or (par[-1] == '3' and J3 > 0):
                hess_sign[par] = -1
            else:
                hess_sign[par] = 1
    is_min = True
    Args = (inp.J1,J2,J3,ans,DerRange[ans],pars,hess_sign,is_min)
    DataDic = {}
    #
    result = d_e(cf.Sigma,
        args = Args,
        x0 = Pinitial[ans],
        bounds = Bnds[ans],
        popsize = 21,
        maxiter = inp.MaxIter*len(Pinitial[ans]),
        #        disp = True,
        tol = inp.cutoff,
        atol = inp.cutoff,
        updating='deferred' if inp.mp_cpu != 1 else 'immediate',
        workers = inp.mp_cpu     #parallelization
        )
    print("\nNumber of iterations: ",result.nit," / ",inp.MaxIter*len(Pinitial[ans]),'\n')
    Pf = tuple(result.x)
    is_min = False
    Args = (inp.J1,J2,J3,ans,DerRange[ans],is_min)
    try:
        S, E, L, gap = cf.Sigma(Pf,*Args)
    except TypeError:
        print("Not saving, an Hessian sign is not right")
        print("Found values: Pf=",Pf,"\nSigma = ",result.fun)
        print("Time of ans",ans,": ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################
        continue
    #Add 0 values
    conv = cf.IsConverged(Pf,Bnds[ans],S)
    newP = cf.FormatParams(Pf,ans,J2,J3)
    data = [ans,J2,J3,conv,E,S,gap,L]
    for ind in range(len(data)):
        DataDic[header[ind]] = data[ind]
    for ind2 in range(len(newP)):
        DataDic[header[len(data)+ind2]] = newP[ind2]
    #save values
    print(DataDic)
    print("Time of ans",ans,": ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################
    cf.SaveToCsv(DataDic,csvfile)

print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################
