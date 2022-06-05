import numpy as np
import initial_states as ins
from pathlib import Path
import time as T
import multiprocessing
#spin structure function
list_ru = [1.50]#np.arange(np.pi/2-0.4,np.pi/2,0.04)#[np.pi/8,np.pi/6,np.pi/4,np.pi/3,np.pi*3/8]
list_rx = [0]#np.arange(np.pi/18,np.pi/6,np.pi/18)#[0]#,np.pi/8,np.pi/6,np.pi/4,np.pi/3,np.pi*3/8,np.pi/2]
list_rz1 = [-np.pi/3]#np.arange(0,2*np.pi,np.pi/6)#[3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4]
list_rz2 = [np.pi/3]#np.arange(0,2*np.pi,np.pi/6)#[3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4]
list_P = []
for ru in list_ru:
    for rx in list_rx:
        for rz1 in list_rz1:
            for rz2 in list_rz2:
                list_P.append([ru,rx,rz1,rz2,True])
###########################
list_z = [np.pi/2]
list_x = [0]
list_P2 = []
for rz in list_z:
    for rx in list_x:
        list_P2.append([rz,rx,True])
for p in list_P2:
    ins.full_func(p)

exit()
if __name__ == "__main__":
    pool = multiprocessing.Pool()
    start_time = T.perf_counter()
    #res = pool.map(full_func, list_P)
    processes = [pool.apply_async(ins.full_func, args=(P,)) for P in list_P]
    result = [p.get() for p in processes]
    finish_time = T.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
#    print(result)





