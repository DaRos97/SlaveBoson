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


dataDir = '../Data/Data_13/'

for file in os.listdir(dataDir):
    with open(dataDir+file, 'r') as f:
        lines = f.readlines()
    with open(dataDir+file, 'w') as f:
        for i in range(8):
            f.write(lines[i])
