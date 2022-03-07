import csv
from pandas import read_csv
import inputs as inp
from colorama import Fore

ans = 0
filename = inp.csvfile[ans]
data = read_csv(filename)

ln = len(data['J2'])

cnt = 0

for i in range(ln):
    S = data['Sigma'][i]
    L = data['L'][i]
    if S > inp.cutoff or L < 0.1:
        print(Fore.RED,data['J2'][i],data['J3'][i],'\t\t',data['Sigma'][i],'\t',data['L'][i],Fore.RESET)
        cnt += 1
    else:
        print(data['J2'][i],data['J3'][i],'\t\t',data['Sigma'][i],'\t',data['L'][i])

print(cnt,ln)
