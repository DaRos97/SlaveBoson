import numpy as np
import inputs as inp
import common_functions as cf
from time import time as t
from colorama import Fore
from scipy.optimize import minimize
from pandas import read_csv
import csv
import sys
####### inputs
J1 = inp.J1
J2, J3 = inp.J[int(sys.argv[1])]
print("Evaluating energy of (J2,J3) = (",J2,",",J3,")")         ##############
#######
Ti = t()
for ans in range(1):#len(inp.text_ans)):            ################
    header = inp.header[inp.text_ans[ans]]
    csvfile = inp.dirname+inp.text_ans[ans]+'/J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
    Args = (J1,J2,J3,ans)
    Tti = t()
    Pi = inp.initialPoint[inp.text_ans[ans]]
    DataDic = {}
    result = minimize(lambda x:cf.Sigma(x,Args),
       Pi,
       method = 'Nelder-Mead',
       bounds = inp.Bnds[inp.text_ans[ans]],
       options = {
           'maxiter':75*len(Pi),
           'fatol':inp.cutoff,
           'adaptive':True}
       )
    Pf = tuple(result.x)
    E,L = cf.totE(Pf,Args)
    S = result.fun
    ########check
    hessian = cf.checkResult(Pf,Args)
    HessDic = {}
    for i in range(len(Pf)):
        HessDic[header[5+i]] = hessian[i]
#    print("After minimization:\n\tparams ",header[5:]," = ",Pf,
#            "\n\tL = ",L,"\n\tSigma = ",S,"\n\tEnergy = ",E)        ###############
    #save values
    data = [J2,J3,E,S,L]
    for ind in range(len(data)):
        DataDic[header[ind]] = data[ind]
    for ind2 in range(len(Pf)):
        DataDic[header[5+ind2]] = Pf[ind2]
    with open(csvfile,'w') as f:
        writer = csv.DictWriter(f, fieldnames = header)
        writer.writeheader()
        writer.writerow(DataDic)
    with open(csvfile,'a') as f:
        writer = csv.DictWriter(f, fieldnames = header[5:])
        writer.writeheader()
        writer.writerow(HessDic)
    if S<inp.cutoff:
        print("Point (",'{:5.5f}'.format(J2),',','{:5.5f}'.format(J3),") of ansatz ",inp.text_ans[ans]," successful")
    else:
        print("Point (",'{:5.5f}'.format(J2),',','{:5.5f}'.format(J3),") of ansatz ",inp.text_ans[ans]," DID NOT CONVERGE")
    print("Time of ans point: ",(t()-Tti)/60,)              ################

print("Total time: ",(t()-Ti)/60)                           ################
