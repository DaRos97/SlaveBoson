import numpy as np
import functions as fs
import inputs as inp
from time import time as t
from colorama import Fore, Style
from scipy.optimize import minimize, minimize_scalar

Ti = t()
J1 = inp.J1
Bnds = ((0,1),(0,10))      #A, L

#just 3x3 ansatz
Ainitial = 0.3       #initial guess A, L
Ai = Ainitial
L = 0.4
Stay = True
rd = 0
rd2 = 0
reps = 2
while Stay:
    ti = t()
    result = minimize_scalar(lambda x:fs.Sigma(x,L),
        #(Ai,Li),
        method = 'bounded',#'Nelder-Mead',
        bounds=Bnds[0]
        #options={
        #    'adaptive':True}
        )
    Af = result.x
    s = fs.Sigma(Af,L)
    L = fs.getL(Af)
    E = fs.totE(Af,L)
    print(Fore.GREEN+"Sigma = :",s," and energy = ",E,"\nParams = ",Af,L,Fore.RESET)
    if s<inp.cutoff and Af > 0.001:
        print("exiting cicle")
        Stay = False
    elif rd2%reps == reps-1 and Af > 0.001:
        print(Fore.RED+"Changing Pi since we are stuck"+Fore.RESET)
        Ai = Ainitial+0.05*(rd2+1)/reps
        rd2 += 1
        print("Starting new cicle with Ai = ",Ai," and L = ",L)
    elif s>inp.cutoff:
        Ai = Af
        print("Starting new cicle with Ai = ",Ai," and L = ",L)
        rd2 += 1
    else:
        print("arrived at A = 0, try again")
        rd += 1
        Ai = Ainitial+0.05*rd
    print("time: ",t()-ti)
    exit()
### print messages
print("Exited minimization with Sigma = ",s)
print("Initial guess: ",Ai,", L = ",L)
print(Fore.GREEN+"Final parameters: ",Af,", L = ",L,Style.RESET_ALL)
print("Energy : ",E)

print("Expected energy: ",-2.203/2)

print(Fore.GREEN+"Total time: ",t()-Ti,Style.RESET_ALL)
