import inputs as inp
import numpy as np
from typing import List, Dict
from pandas import read_csv

header = inp.header
data = read_csv(inp.csvfile[0])
J3 = data['Energy'].tolist()
print(J3)
