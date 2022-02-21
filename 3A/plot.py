import numpy as np
import inputs as inp
import matplotlib.pyplot as plt

#read data from files
S = 0.5
pts = int(input("number of points: "))
dirname = 'DataS'+str(S).replace('.','')+'/'
Jmin = -0.3
Jmax = 0.3
Jr = Jmax-Jmin
dataE = np.ndarray((2,pts,pts))
dataS = np.ndarray((2,pts,pts))
dataP = np.ndarray((2,3,pts,pts))
dataL = np.ndarray((2,2,pts,pts))
Dtxt = inp.text_params
for ans in range(2):
    dataE[ans] = np.load(dirname+Dtxt[0]+'_(J2,J3)-'+inp.text_ans[ans]+'PDpts='+str(int(pts))+'a.npy')
    dataS[ans] = np.load(dirname+Dtxt[1]+'_(J2,J3)-'+inp.text_ans[ans]+'PDpts='+str(int(pts))+'a.npy')
    dataP[ans] = np.load(dirname+Dtxt[2]+'_(J2,J3)-'+inp.text_ans[ans]+'PDpts='+str(int(pts))+'a.npy')
    dataL[ans] = np.load(dirname+Dtxt[3]+'_(J2,J3)-'+inp.text_ans[ans]+'PDpts='+str(int(pts))+'a.npy')
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
JM = Jmax + Jr/(pts-1)/2
Jm = Jmin - Jr/(pts-1)/2
for i in range(pts+1):
    plt.plot((Jm+Jr/(pts-1)*i,Jm+Jr/(pts-1)*i),(Jm,JM),'k')
    plt.plot((Jm,JM),(Jm+Jr/(pts-1)*i,Jm+Jr/(pts-1)*i),'k')
for i in range(pts):
    for j in range(pts):
        plt.fill_between(
                np.linspace(Jm+Jr/(pts-1)*i,Jm+Jr/(pts-1)*(i+1),10),
                np.linspace(Jm+Jr/(pts-1)*(j+1),Jm + Jr/(pts-1)*(j+1),10),
                Jm+Jr/(pts-1)*j,
                color=Color[Ans[i,j]+SL[i,j]],
                label= Label[Ans[i,j]+SL[i,j]])
for i in range(pts):
    for j in range(pts):
        plt.scatter(Jmin+Jr/(pts-1)*i,Jmin+Jr/(pts-1)*j,marker='.',color = 'k')
for i in range(4):
    plt.text(JM+Jr/(pts-1)/2,Jmax-Jr/(pts-1)*i,Label[i],color=Color[i])

plt.xlabel("$J_2$",size=20)
plt.ylabel("$J_{3e}$",size=20)
plt.show()
