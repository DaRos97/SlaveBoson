import numpy as np
import functions as fs
import inputs as inp
from time import time as t
from colorama import Fore, Style
from scipy.optimize import minimize, minimize_scalar

dirname = 'DataS'+str(inp.S).replace('.','')+'AB/'
Ti = t()
J1 = inp.J1
Bnds = ((0,1),(0,1),(0,1))      #A,B,L

initialP = (0.26,0.05,0.4)       #initial guess from classical values??
ti = t()
print("Initial guess: ",initialP)
L = 0.4
result = minimize(lambda x:fs.SigmaLab((x[0],x[1],x[2])),
    initialP,
    method = 'Powell',
    bounds=Bnds,
    tol = 1e-8,
    options={'disp':True,
        'xtol':1e-4})
Pf = result.x
#L = fs.Tot_Eab((Pf[0],Pf[1],L))[1]
e = fs.E3Lab(Pf)
### print messages
print("Energy of this step: ",e)
print("with parameters: ",Pf)
print("Time of this point: ",t()-ti)




print(Fore.BLUE+"Total time: ",t()-Ti,Style.RESET_ALL)
