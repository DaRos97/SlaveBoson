import numpy as np
import matplotlib.pyplot as plt

S = 0.5
J2 = 0.2
text_ans = ['(0,0)','(pi,0)','(pi,pi)','(0,pi)']
colors = ['b*','r*','m*','g*']
fig = plt.figure(figsize=(12,8))
dirname = ['data2nn/']
for i in range(1):
    text = dirname[0] + text_ans[1] + 'E_2nn-J2=' + str(J2).replace('.',',') + 'phi1.npy'
    data = np.load(text)
    plt.plot(data[0],data[1],colors[i],label=text_ans[i])
plt.title('S = '+str(S))
plt.xlabel(r"$\phi$ of DM")
plt.ylabel(r"$\frac{E}{S(S+1)}$")
plt.legend()
plt.show()

