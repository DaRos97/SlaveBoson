import numpy as np
import functions as fs
import inputs as inp
from time import time as t
from colorama import Fore, Style
from scipy.optimize import minimize, minimize_scalar

dirname = 'Data1'
ti = t()
J1 = inp.J1
Bnds = ((0,1),(0,1),(0,1))      #should make a function of this

# 3x3 PD
ans = 0
for ans in range(2):
    initialP = np.array([0.14,0.43,0.46])       #initial guess from classical values??
    E_arr = np.zeros((inp.PD_pts,inp.PD_pts))
    S_arr = np.zeros((inp.PD_pts,inp.PD_pts))
    print(Fore.GREEN+"Using ansatz ",inp.text_ans[ans])
    for j2,J2 in enumerate(inp.rJ2[ans]):
        for j3,J3 in enumerate(inp.rJ3[ans]):
            ti = t()
            print(Fore.RED+"Evaluating energy of (j2,j3)=(",J2,",",J3,") point",Style.RESET_ALL)
            print("Initial guess: ",initialP)
            Args = (J1,J2,J3,ans)       #tuple of arguments to pass to Sigma
            result = minimize(lambda x:fs.Sigma(x,Args),
                    initialP,
                    method = 'Powell',
                    bounds=Bnds,
                    options={'disp':True,
                        'xtol':1e-4})
            initialP = result.x
            E_arr[j2,j3] = fs.tot_E(initialP,Args)
            S_arr[j2,j3] = fs.Sigma(initialP,Args)
            print("Sigma of this step: ",S_arr[j2,j3])
            print("Energy of this step: ",E_arr[j2,j3])
            print("with parameters: ",initialP)
            print("Time of this point: ",t()-ti)
    #if sigma>0 --> LRO
    #if sigma~0 --> SL
    textE = dirname+'Energies_(J2,J3)-'+inp.text_ans[ans]+'pts=',str(int(inp.PD_pts))+'.npy'
    textS = dirname+'Sigmas_(J2,J3)-'+inp.text_ans[ans]+'pts=',str(int(inp.PD_pts))+'.npy'
    np.save(textE,E_arr)
    np.save(textS,S_arr)
