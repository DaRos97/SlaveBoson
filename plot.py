import numpy as np
import matplotlib.pyplot as plt

S = 0.5
text_ans = ['(0,0)','(pi,0)','(pi,pi)','(0,pi)']
colors = ['b*','r*','m*','g*']
colors2 = ['y*','k*','r*','b*']
fig = plt.figure(figsize=(12,8))
dirname = ['data/','data3/']
plt.subplot(1,2,1)
for i in range(4):
    text = dirname[0] + text_ans[i] + 'E_gs-' + str(S).replace('.',',') + '.npy'
    data = np.load(text)
    plt.plot(data[0],data[1],colors[i],label=text_ans[i])
plt.title('S = '+str(S))
plt.xlabel(r"$\phi$ of DM")
plt.ylabel(r"$\frac{E}{S(S+1)}$")
plt.legend()
plt.subplot(1,2,2)
for i in range(4):
    text = dirname[1] + text_ans[i] + 'E_gs-' + str(S).replace('.',',') + '.npy'
    data = np.load(text)
    plt.plot(data[0],data[1],colors[i],label=text_ans[i])

plt.xlabel(r"$\phi$ of DM")
plt.ylabel(r"$\frac{E}{S(S+1)}$")
plt.legend()
plt.show()

