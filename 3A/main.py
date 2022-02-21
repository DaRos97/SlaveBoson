import numpy as np
import functions as fs
import inputs as inp
from time import time as t
from colorama import Fore, Style
from scipy.optimize import minimize, minimize_scalar

dirname = 'DataS'+str(inp.S).replace('.','')+'/'
Ti = t()
J1 = inp.J1
Bnds = ((0,1),(0,1),(0,1))      #A1,A2,A3

for ans in range(2):
    Pinitial = (0.5,0.3,0.2)       #initial guess of A1,A2,A3 from classical values?? see art...
    Pi = Pinitial
    E_arr = np.zeros((inp.PD_pts,inp.PD_pts))
    P_arr = np.zeros((3,inp.PD_pts,inp.PD_pts))
    S_arr = np.zeros((inp.PD_pts,inp.PD_pts))
    L_arr = np.zeros((inp.PD_pts,inp.PD_pts))
    print(Fore.GREEN+"Using ansatz ",inp.text_ans[ans])
    for j2,J2 in enumerate(inp.rJ2):
        for j3,J3 in enumerate(inp.rJ3):
            Tti = t()
            print(Fore.RED+"Evaluating energy of (J2,J3) = (",J2,",",J3,")",Style.RESET_ALL)
            Stay = True
            rd = 0
            rd2 = 0
            while Stay:
                ti = t()
                print("Initial guess: ",Pi)
                Args = (J1,J2,J3,ans)
                result = minimize(lambda x:fs.Sigma(x,Args)[0],
                    Pi,
                    method = 'Nelder-Mead',
                    bounds=Bnds,
                    options={'disp':False,
                        'adaptive':True}
                    )
                Pf = result.x
                #checks
                S = fs.Sigma(Pf,Args)
                E,L = fs.totE(Pf,Args)
                print("After minimization:\n\tparams = ",Pf,"\n\tL = ",L,"\n\tSigma = ",S,"\n\tEnergy = ",E)
                if S<inp.cutoff:
                    print("exiting cicle")
                    Stay = False
                    #save values
                    Pi = Pf
                    P_arr[:,j2,j3] = Pf
                    E_arr[j2,j3] = E
                    S_arr[j2,j3] = S
                    L_arr[j2,j3] = L
                elif S>inp.cutoff and rd2%2 == 0:
                    Pi = Pf
                    print("Sigma not small enough, starting new cicle with Pi = ",Pi)
                    rd2 += 1
                else:
                    rd += 1
                    for i in range(3):
                        Pi[i] = Pinitial[i] + 0.05*rd
                    print(Fore.Blue+"Changing initial parameters to ",Pi)
                print(Fore.YELLOW+"time of while cicle: ",t()-ti,Fore.RESET)
            print(Fore.YELLOW+"time of (j2,j3) point: ",t()-Tti,Fore.RESET)
            exit()
    #save externally
    textE = dirname+'Energies_(J2,J3)-'+inp.text_ans[ans]+'PDpts='+str(int(inp.PD_pts))+'.npy'
    textS = dirname+'Sigmas_(J2,J3)-'+inp.text_ans[ans]+'PDpts='+str(int(inp.PD_pts))+'.npy'
    np.save(textE,E_arr)
    np.save(textS,S_arr)

print(Fore.BLUE+"Total time: ",t()-Ti,Style.RESET_ALL)
