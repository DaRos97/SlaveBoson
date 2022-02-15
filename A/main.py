import numpy as np
import functions as fs
import inputs as inp
from time import time as t
from colorama import Fore, Style
from scipy.optimize import minimize

ti = t()

for j2,J2 in enumerate(inp.rJ2):
    for j3,J3 in enumerate(inp.rJ3):
        Args = [inp.J1,J2,J3]
        initialP = [0.2,0.1,0.01,0.5]
        Bnds = ((0,0.5),(0,0.5),(0,0.5),(0,1))
        result = minimize(fs.Sigma,
                    initialP,
                    args=Args,
                    method = 'Powell',
                    options={'xtol':1e-4})
        print(result.x)

