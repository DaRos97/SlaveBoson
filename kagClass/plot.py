import matplotlib.pyplot as plt
import numpy as np
import initial_states as ins

ru = 1.4#np.pi/4
rx = 0#np.pi/7
rz1 = np.pi/3
rz2 = -np.pi/3
P = [ru,rx,rz1,rz2,True]
ins.full_func(P)

exit()
