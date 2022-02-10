import numpy as np
import matplotlib.pyplot as plt
import inputs_2nn as inp
from pathlib import Path

S = 0.5
text_ans = ['(0,0)','(pi,0)','(pi,pi)','(0,pi)']
colors = ['b*-','r*-','m*-','g*-']
fig = plt.figure(figsize=(12,8))
dirname = ['Data/','Data2/','Data3/']

ans = 0
plt.subplot(1,2,1)
for j3,J3 in enumerate(np.linspace(0,inp.J3_max,inp.J3_pts)):
    path_alpha = Path('alphaD/alphas-'+inp.text_ans[ans]+'_J3='+"{:.4f}".format(J3).replace('.',',')+'.npy')
    data = np.load(path_alpha)
    plt.plot(data[0],data[1],label='J3='+"{:.4f}".format(J3))
plt.legend()
ans = 1
plt.subplot(1,2,2)
for j2,J2 in enumerate(np.linspace(0,inp.J2_max,inp.J2_pts)):
    path_alpha = Path('alphaD/alphas-'+inp.text_ans[ans]+'_J2='+"{:.4f}".format(J2).replace('.',',')+'.npy')
    data = np.load(path_alpha)
    plt.plot(data[0],data[1],label='J2='+"{:.4f}".format(J2))
plt.legend()


#plt.subplot(1,2,2,sharey = ax1)
#for i in range(4):
#    text = dirname[1] + text_ans[i] + 'E_gs-' + str(S).replace('.',',') + '.npy'
#    data = np.load(text)
#    plt.plot(data[0],data[1],colors[i],label=text_ans[i])
#plt.title('S = '+str(S))
#plt.xlabel(r"$\phi$ of DM")
#plt.legend()


plt.show()

