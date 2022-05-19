import numpy as np
import functions as fs
import inputs as inp
from time import time as t
from colorama import Fore, Style
from scipy.optimize import minimize, minimize_scalar

dirname = 'DataS02/'
Ti = t()
J1 = inp.J1
Bnds = ((0,(2*inp.S+1)/2),(0,2*inp.S))      #A,B

#just 3x3 ansatz
Pinitial = np.array([0.1,0.01])
Pi = Pinitial       #initial guess A,B
Stay = True
rd = 0
rd2 = 0
while Stay:
    ti = t()
    result = minimize(lambda x:fs.Sigma(x),
        Pi,
        method = 'Nelder-Mead',#'L-BFGS-B',
        bounds=Bnds,
        options={
            'adaptive':True}
        )
    Pf = result.x
    s = fs.Sigma(Pf)
    L = fs.getL(Pf)
    print(Fore.RED+"Parameters found: ",Pf,Fore.RESET)
    if s<inp.cutoff and Pf[0]+Pf[1] > 0.01:
        print("exiting cicle")
        Stay = False
    elif rd2%3 == 2 and Pf[0]+Pf[1] > 0.01:
        print("changing Pi since we are stuck")
        Pi[0] = Pinitial[0]+0.05*rd2
        Pi[1] = Pinitial[1]+0.01*rd2
        rd2 += 1
        print("Starting new cicle with Pi = ",Pi,L)
    elif s>inp.cutoff:
        Pi = Pf
        print("Sigma = :",s)
        print("Starting new cicle with Pi = ",Pi,L)
        rd2 += 1
    else:
        print("arrived at P = 0, try again")
        Pi[rd%2] = Pinitial[rd%2]+0.05*(rd%2)+0.1*((rd+1)%2)
        Pi[(rd+1)%2] = Pinitial[(rd+1)%2]
        rd += 1
    print("time: ",t()-ti)

E = fs.totE(Pf,L)
### print messages
print("Sigma = ",s)
print("Initial guess: ",Pi,", L = nan")
print(Fore.GREEN+"Final parameters: ",Pf,", L = ",L,Style.RESET_ALL)
print("Energy : ",E)

print("Expected energy: ",fs.totE([0.2637,0.0574],0.4182))

print(Fore.GREEN+"Total time: ",t()-Ti,Style.RESET_ALL)
