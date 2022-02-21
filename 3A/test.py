import csv
import inputs as inp
import numpy as np

header = ['J2','J3','Energy','Sigma','A1','A2','A3','L','minimum L']
E = np.load(inp.text_file[0][0])
S = np.load(inp.text_file[0][1])
P = np.load(inp.text_file[0][2])
L = np.load(inp.text_file[0][3])
#data = [E.T,S.T,P[0].T,P[1].T,P[2].T,L[0].T,L[1].T]

data = [-0.3,-0.3,-0.66,1e-10,0.52,0.,0.38,1.8234,1.8233]

with open('csvtest.csv','r',encoding='UTF8',newline='') as f:
    reader = csv.reader(f)
    #writer.writerow(header)
    print(len(list(reader)))
    for row in reader:
        

exit()
with open('csvtest.csv','r',encoding='UTF8') as f:
    reader = csv.reader(f)
    b = np.ndarray(0)
    for row in reader:
        print(row)
        b = np.append(b,float(row[0]))

print(b)

