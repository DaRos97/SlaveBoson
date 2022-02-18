import numpy as np
import inputs as inp
import matplotlib.pyplot as plt

S = 0.2
pts = 5
dirname = 'DataS'+str(S).replace('.','')+'/'
Jmax = 0.3
dataE = np.ndarray((2,pts,pts))
dataS = np.ndarray((2,pts,pts))
for ans in range(2):
    dataE[ans] = np.load(dirname+'Energies_(J2,J3)-'+inp.text_ans[ans]+'pts='+str(int(pts))+'a.npy')
    dataS[ans] = np.load(dirname+'Sigmas_(J2,J3)-'+inp.text_ans[ans]+'pts='+str(int(pts))+'a.npy')

E = np.zeros((pts,pts),dtype=int)
S = np.zeros((pts,pts))
for i in range(pts):
    for j in range(pts):
        E[i,j] = 0 if dataE[0,0,j] < dataE[1,i,0] else 2 
        S[i,j] = dataS[0,0,j] if E[i,j] == 0 else dataS[1,i,0]
        if S[i,j] < 1e-4:
            S[i,j] = 1
        else:
            S[i,j] = 0
        E[i,j] += S[i,j]

Color = ['orange','r','c','b']
Label = ['(0,0)-LRO','(0,0)-SL','(0,pi)-LRO','(0,pi)-SL']
plt.figure(figsize=(10,8))
for i in range(pts+1):
    plt.plot((Jmax/pts*i,Jmax/pts*i),(0,Jmax),'k')
    plt.plot((0,Jmax),(Jmax/pts*i,Jmax/pts*i),'k')
for i in range(pts):
    for j in range(pts):
        plt.fill_between(
                np.linspace(Jmax/pts*i,Jmax/pts*i+Jmax/pts,10),
                np.linspace(Jmax/pts*j+Jmax/pts,Jmax/pts*j+Jmax/pts,10),
                Jmax/pts*j,
                color=Color[E[i,j]],
                label= Label[E[i,j]])
for i in range(4):
    plt.text(Jmax+0.02,Jmax-Jmax/pts*i*0.3,Label[i],color=Color[i])

plt.xlabel("$J_2$",size=20)
plt.ylabel("$J_{3e}$",size=20)
plt.show()
