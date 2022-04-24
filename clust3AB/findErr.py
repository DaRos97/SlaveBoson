import numpy as np
import inputs as inp
import os

dirname = '../Data/yesDMsmall/Data_17/'
cnt = 0
for i in range(3):
    for filename in os.listdir(dirname):
        with open(dirname+filename, 'r') as f:
            lines = f.readlines()
        data = lines[i*4+1].split(',')
        a = False
        if np.abs(float(data[6])) < 0.51 and np.abs(float(data[6])) > 0.5:
            cnt += 1
            print("Min at: \tans=",data[0],"\t(J2,J3)=","{:5.4f}".format(float(data[1])),"{:5.4f}".format(float(data[2])))
print(cnt)
