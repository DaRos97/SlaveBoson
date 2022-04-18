import numpy as np
import inputs as inp
import common_functions as cf
from time import time as t
from colorama import Fore
from scipy.optimize import minimize
from pandas import read_csv
import csv
import sys
import os

import findDomain as fD
####### inputs
J1 = inp.J1
J2, J3 = inp.J[int(sys.argv[1])]
print('(J2,J3) = ('+'{:5.4f}'.format(J2)+',{:5.4f}'.format(J3)+')\n')
#######
ansatze = ['3x3']
Bnds = cf.findBounds(J2,J3,ansatze)
args = (J1,J2,J3,'3x3')
res = fD.Domain(Bnds['3x3'],args)

print(res)
