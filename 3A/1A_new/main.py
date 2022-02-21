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
Linitial = 0.4
L = Linitial
Stay = True
rd = 0
rd2 = 0
reps = 4
while Stay:
    print("Starting new cicle with Ai = ",Ai," and L = ",L)
    ti = t()
    result = minimize_scalar(lambda x:fs.Sigma(x),#fs.totE(x),#fs.Sigma(x)[0],
        #(Ai,L),
        method = 'bounded',#'Nelder-Mead',#'bounded',#
        bounds = Bnds[0],
        options={
            'xatol':1e-5
        #    'adaptive':True}
        })
    Af = result.x
    L = fs.minL(Af)
    print(Fore.RED,Af,L,fs.totE(Af),Fore.RESET)
    s = fs.Sigma(Af)
    print(Fore.BLUE,Af,L,s,Fore.RESET)
    #L = fs.getL(Af)
    E = fs.totE(Af)
    print(Fore.GREEN+"Sigma = :",s," and energy = ",E,"\nParams = ",Af,L,Fore.RESET)
    Ai = Af
    if s<inp.cutoff and Af > 0.001:
        print("exiting cicle")
        Stay = False
    elif rd2%reps == reps-1 and Af > 0.001:
        print(Fore.RED+"Changing Pi since we are stuck"+Fore.RESET)
        Ai = Ainitial+0.05*(rd2+1)/reps
        rd2 += 1
    elif s>inp.cutoff:
        Ai = Af
        rd2 += 1
    else:
        print(Fore.BLUE,"arrived at A = 0, try again",Fore.RESET)
        rd += 1
        Ai = Ainitial+0.05*rd
    print("time: ",t()-ti)
### print messages
print('\n')
print("Exited minimization with Sigma = ",s)
print("Initial guess: ",Ainitial,", L = ",Linitial)
print(Fore.GREEN+"Final parameters: ",Af,", L = ",L,Style.RESET_ALL)
print("Energy : ",E)

print("Expected energy: ",-2.203/2)

print(Fore.GREEN+"Total time: ",t()-Ti,Style.RESET_ALL)
