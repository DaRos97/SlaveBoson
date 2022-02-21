import numpy as np
import inputs as inp
import matplotlib.pyplot as plt

#read data from files
S = 0.5
pts = 2
dirname = 'DataS'+str(S).replace('.','')+'/'
Jmax = 0.3
dataE = np.ndarray((2,pts,pts))
dataS = np.ndarray((2,pts,pts))
dataP = np.ndarray((2,3,pts,pts))
dataL = np.ndarray((2,2,pts,pts))
Dtxt = inp.text_params
for ans in range(2):
    dataE[ans] = np.load(dirname+Dtxt[0]+'_(J2,J3)-'+inp.text_ans[ans]+'PDpts='+str(int(pts))+'.npy')
    dataS[ans] = np.load(dirname+Dtxt[1]+'_(J2,J3)-'+inp.text_ans[ans]+'PDpts='+str(int(pts))+'.npy')
    dataP[ans] = np.load(dirname+Dtxt[2]+'_(J2,J3)-'+inp.text_ans[ans]+'PDpts='+str(int(pts))+'.npy')
    dataL[ans] = np.load(dirname+Dtxt[3]+'_(J2,J3)-'+inp.text_ans[ans]+'PDpts='+str(int(pts))+'.npy')
#organize energies
E = np.zeros((pts,pts),dtype=int)
Ans = np.zeros((pts,pts),dtype=int)
SL = np.zeros((pts,pts),dtype=int)
for i in range(pts):
    for j in range(pts):
        if dataE[0,i,j] < dataE[1,i,j]:
            E[i,j] = dataE[0,i,j]
            Ans[i,j] = 0
            SL[i,j] = 0 if np.abs(dataL[0,0,i,j]-dataL[0,1,i,j]) < 1e-3 else 1
        else:
            E[i,j] = dataE[1,i,j]
            SL[i,j] = 0 if np.abs(dataL[1,0,i,j]-dataL[1,1,i,j]) < 1e-3 else 1
            Ans[i,j] = 2

Color = ['orange','r','c','b']
Label = ['(0,0)-LRO','(0,0)-SL','(0,pi)-LRO','(0,pi)-SL']
plt.figure(figsize=(10,8))
#grid
for i in range(pts+1):
    plt.plot((Jmax/pts*i,Jmax/pts*i),(0,Jmax),'k')
    plt.plot((0,Jmax),(Jmax/pts*i,Jmax/pts*i),'k')
for i in range(pts):
    for j in range(pts):
        plt.fill_between(
                np.linspace(Jmax/pts*i,Jmax/pts*i+Jmax/pts,10),
                np.linspace(Jmax/pts*j+Jmax/pts,Jmax/pts*j+Jmax/pts,10),
                Jmax/pts*j,
                color=Color[Ans[i,j]+SL[i,j]],
                label= Label[Ans[i,j]+SL[i,j]])
for i in range(4):
    plt.text(Jmax+0.02,Jmax-Jmax/pts*i*0.3,Label[i],color=Color[i])

plt.xlabel("$J_2$",size=20)
plt.ylabel("$J_{3e}$",size=20)
plt.show()
