import numpy as np
import functions as fs
import inputs as inp
from time import time as t
from colorama import Fore, Style
from scipy.optimize import minimize, minimize_scalar

Ti = t()
J1 = inp.J1
Bnds = ((0,1),(0,1),(0,1))      #A1,A2,A3
Pinitial = (0.5,0.1,0.1)       #initial guess of A1,A2,A3 from classical values?? see art...
non_converging_points = 0
reps = 3
for ans in range(2):
    Ttti = t()
    Pi = Pinitial
    E_arr = np.zeros((inp.PD_pts,inp.PD_pts))
    S_arr = np.zeros((inp.PD_pts,inp.PD_pts))
    P_arr = np.zeros((3,inp.PD_pts,inp.PD_pts))
    L_arr = np.zeros((2,inp.PD_pts,inp.PD_pts))
    print(Fore.GREEN+"\nUsing ansatz ",inp.text_ans[ans])
    for j2,J2 in enumerate(inp.rJ2[ans]):
        for j3,J3 in enumerate(inp.rJ3[ans]):
            Tti = t()
            print(Fore.RED+"\nEvaluating energy of (J2,J3) = (",J2,",",J3,")",Style.RESET_ALL)
            Stay = True
            rd = 0
            rd2 = 0
            tempE = []; tempS = []; tempP = []; tempL = []; tempmL = [];
            while Stay:
                ti = t()
                print("Initial guess: ",Pi)
                Args = (J1,J2,J3,ans)
                result = minimize(lambda x:fs.Sigma(x,Args),
                    Pi,
                    method = 'Nelder-Mead',
                    bounds = Bnds,
                    options = {
                        'adaptive':True}
                    )
                Pf = result.x
                #checks
                Pf[ans+1] = 0
                if J3 == 0:
                    Pf[2] == 0
                if J2 == 0:
                    Pf[1] == 0
                S = fs.Sigma(Pf,Args)
                E,L,mL = fs.totE(Pf,Args)
                print("After minimization:\n\tparams = ",Pf,"\n\tL,mL = ",L,mL,"\n\tSigma = ",S,"\n\tEnergy = ",E)
                if S<inp.cutoff:
                    print("exiting cicle")
                    Stay = False
                    #save values
                    Pi = Pf
                    E_arr[j2,j3] = E
                    S_arr[j2,j3] = S
                    P_arr[:,j2,j3] = Pf
                    L_arr[:,j2,j3] = [L,mL]
                elif rd <= reps:
                    tempE += [E]
                    tempS += [S]
                    tempP += [Pf]
                    tempL += [L]
                    tempmL += [mL]
                    rd += 1
                    Pi = (Pinitial[0]+0.05*rd,Pinitial[1]+0.05*rd,Pinitial[2]+0.05*rd)
                    print(Fore.BLUE+"Changing initial parameters to ",Pi,Fore.RESET)
                else:
                    print(Fore.GREEN+"It's taking too long, pass to other point")
                    Stay = False
                    arg = np.argmin(tempE)
                    E_arr[j2,j3] = tempE[arg]
                    S_arr[j2,j3] = tempS[arg]
                    P_arr[:,j2,j3] = tempP[arg]
                    L_arr[:,j2,j3] = [tempL[arg],tempmL[arg]]
                    print("Keeping the best result:\n\tparams = ",P_arr[:,j2,j3],"\n\tL,mL = ",L_arr[:,j2,j3],"\n\tSigma = ",S_arr[j2,j3],"\n\tEnergy = ",E_arr[j2,j3],Fore.RESET)
                    non_converging_points += 1
            print(Fore.YELLOW+"time of (j2,j3) point: ",t()-Tti,Fore.RESET)
    #compute missing points
    for i in range(1,inp.PD_pts):
        for j in range(inp.PD_pts):
            if ans == 0:
                E_arr[i,j] = E_arr[0,j] + inp.rJ[ans][1-ans][i]*inp.z[ans+1]*inp.S**2/2
            else:
                E_arr[j,i] = E_arr[j,0] + inp.rJ[ans][1-ans][i]*inp.z[ans+1]*inp.S**2/2
    #save externally
    Data = [E_arr,S_arr,P_arr,L_arr]
    if inp.Save:
        print(Fore.BLUE+"Saving values in ",inp.dirname,Fore.RESET)
        for i in range(len(inp.text_params)):
            np.save(inp.text_file[ans][i],Data[i])
    print(Fore.YELLOW+"time of ansatz: ",t()-Ttti,Fore.RESET)
print(Fore.GREEN+"Non converging points: ",non_converging_points,Fore.RESET)
print(Fore.YELLOW+"Total time: ",t()-Ti,Style.RESET_ALL)
