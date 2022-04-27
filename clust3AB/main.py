import numpy as np
import inputs as inp
import common_functions as cf
from time import time as t
from scipy.optimize import differential_evolution as d_e
from pandas import read_csv
import csv
import sys
import os
####### inputs
J1 = inp.J1
J2, J3 = inp.J[int(sys.argv[1])]
print('\n(J2,J3) = ('+'{:5.4f}'.format(J2)+',{:5.4f}'.format(J3)+')\n')
#######
csvfile = inp.DataDir+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
#ansatze = cf.CheckCsv(csvfile)
ansatze = inp.list_ans
Ti = t()
Pinitial = cf.FindInitialPoint(J2,J3,ansatze)
Bnds = cf.FindBounds(J2,J3,ansatze)
DerRange = cf.ComputeDerRanges(J2,J3,ansatze)
for ans in ansatze:
    Tti = t()
    print("Using ansatz: ",ans)
    header = inp.header[ans]
    Pi = Pinitial[ans]
    bnds = Bnds[ans]
    der_range = DerRange[ans]
    Args1 = (J1,J2,J3,ans,der_range)
    Args = (J1,J2,J3,ans)
    DataDic = {}
    HessDic = {}
    print("Initial point and bounds: \n",Pi,'\n',bnds,'\n')
    result = d_e(cf.Sigma2,
        args = Args1,
        x0 = Pi,
        bounds = bnds,
        popsize = 15,
        maxiter = inp.MaxIter,
        disp = False,
        tol = inp.cutoff,
        atol = inp.cutoff,
        updating='deferred' if inp.mp_cpu != 1 else 'immediate',
        workers = inp.mp_cpu     #parallelization
        )
    print("\nNumber of iterations: ",result.nit," / ",inp.MaxIter,'\n')
    Pf = tuple(result.x)
    S = result.fun
    E,L = cf.totE(Pf,Args)[:2]
    #Add 0 values
    newP = cf.arangeP(Pf,ans,J2,J3)
    ########Compute Hessian values
    hessian = cf.arangeP(cf.Hessian(Pf,Args1),ans,J2,J3)
    for i in range(len(hessian)):
        HessDic[header[6+i]] = hessian[i]
    #save values
    data = [ans,J2,J3,E,S,L]
    for ind in range(len(data)):
        DataDic[header[ind]] = data[ind]
    for ind2 in range(len(newP)):
        DataDic[header[6+ind2]] = newP[ind2]
    print(DataDic)
    print(HessDic)
    print("Time of ans",ans,": ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################
    cf.SaveToCsv(DataDic,HessDic,csvfile)

print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################
