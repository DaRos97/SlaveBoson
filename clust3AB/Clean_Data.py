import numpy as np
import os

dirname = '../Data_11/'
newdir = 'Data/Final_11/'

save_ans = ['3x3', 'q0']
#save everything
dic = {}
for ans in save_ans:
    dic[ans] = []

for filename in os.listdir(dirname):
    with open(dirname+filename, 'r') as f:
        lines = f.readlines()
    N = (len(lines)-1)//4 + 1
    for i in range(N):
        data = lines[i*4+1].split(',')
        if data[0] in save_ans:
            tDic = {'ans':data[0]}
            head = lines[i*4].split(',')
            for j in range(1,len(data)):
                tDic[head[j]] = float(data[j])
            dic[data[0]].append(tDic)
    J2 = float(filename[7:-5].split('_')[0])/10000  #specific for the name of the file
    J3 = float(filename[7:-5].split('_')[1])/10000
    newcsv = newdir+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
    with open(newdir+newcsv, 'a'):
            writer = csv.DictWriter(f, fieldnames = header)
            writer.writeheader()
            writer.writerow(Data)
            writer = csv.DictWriter(f, fieldnames = header[6:])
            writer.writeheader()
            writer.writerow(Hess)

