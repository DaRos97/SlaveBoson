import numpy as np
import functions as fs
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

ans = '3x3_1'
DM_ = np.pi/8
DM = "{:3.2f}".format(DM_).replace('.','')
pts = '13'
S = 0.5
dL = 1e-6
savefile = ans+'-'+DM+'.npy'

dirname = '../Data/phi'+DM+'/'+pts+'/'
cond = np.zeros((9,9))
if not Path(savefile).is_file():
    for filename in os.listdir(dirname):
        params = fs.import_data(ans,dirname+filename)
        if len(params) == 0:
            continue
        if params[3] != 'True':
            continue
        #calc two times omega, with dL
        w0 = fs.compute_w(params,ans,DM_)
        params[7] += dL
        w1 = fs.compute_w(params,ans,DM_)
        params[7] += dL
        w2 = fs.compute_w(params,ans,DM_)
        #
        der0 = (w1-w0)/dL
        der1 = (w2-w1)/dL
        hess = (der1-der0)/dL
        if hess > 0:
            print("Hessian of (","{:5.4f}".format(params[1]),",","{:5.4f}".format(params[2]),") is: ",hess)
        J2 = params[1]
        J3 = params[2]
        i2 = int((float(J2)+0.3)*8/0.6)
        i3 = int((float(J3)+0.3)*8/0.6)
        cond[i3,i2] = -der0+(2*S+1)
    for i in range(9):
        for j in range(9):
            if cond[i,j] == 0:
                cond[i,j] = np.nan
    np.save(savefile, cond)
else:
    cond = np.load(savefile)

#plot
fig = plt.figure()
plt.axis('off')
ax = fig.add_subplot(111,projection='3d')
X,Y = np.meshgrid(np.linspace(-0.3,0.3,9),np.linspace(-0.3,0.3,9))
ax.plot_surface(X,Y,cond,cmap=cm.plasma)
plt.show()
