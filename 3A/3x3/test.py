import inputs as inp
import numpy as np
from typing import List, Dict
from pandas import read_csv

header = inp.header
data = read_csv(inp.csvfile[0],usecols=['J2','J3'])
a = data['J2']
b = data['J3']
print(b)

