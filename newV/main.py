import numpy as np
import inputs as inp
import common_functions as cf
from time import time as t
from scipy.optimize import differential_evolution as d_e
import sys
####### inputs
li = [4,36,40,44,76]
J2, J3 = inp.J[int(sys.argv[1])]
print('\n(J2,J3) = ('+'{:5.4f}'.format(J2)+',{:5.4f}'.format(J3)+')\n')
#######
csvfile = inp.DataDir+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
print("File name for saving: ",csvfile)
#ansatze = cf.CheckCsv(csvfile)
#ansatze = inp.list_ans
ANS = int(sys.argv[1])//5
ansatze = [inp.list_ans[ANS]]
print('Ansatze: ',ansatze)
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
    Args = (inp.J1,J2,J3,ans,der_range)
    DataDic = {}
    HessDic = {}
    print("Initial point and bounds: \n",Pi,'\n',bnds)
    #
    result = d_e(cf.Sigma,
            args = Args,
            x0 = Pi,
            bounds = bnds,
            popsize = 16,#inp.mp_cpu*2,
            maxiter = inp.MaxIter*len(Pi),
            disp = True,
            tol = inp.cutoff,
            atol = inp.cutoff,
            updating='deferred' if inp.mp_cpu != 1 else 'immediate',
            workers = inp.mp_cpu     #parallelization
            )
    print("\nNumber of iterations: ",result.nit," / ",inp.MaxIter*len(Pi),'\n')
    Pf = tuple(result.x)
    try:
        S, HessVals, E, L, gap = cf.Final_Result(Pf,*Args)
    except TypeError:
        print("Not saving, there was some mistake")
        print("Found values: Pf=",Pf,"\nSigma = ",result.fun)
        continue
    #Add 0 values
    newP = cf.FormatParams(Pf,ans,J2,J3)
    data = [ans,J2,J3,E,S,gap,L]
    for ind in range(len(data)):
        DataDic[header[ind]] = data[ind]
    for ind2 in range(len(newP)):
        DataDic[header[7+ind2]] = newP[ind2]
    #save values
    print(DataDic)
    print(HessVals)
    print("Time of ans",ans,": ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################
    cf.SaveToCsv(DataDic,csvfile)

print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################
