import matplotlib.pyplot as plt
import numpy as np
import inputs as inp
from scipy.interpolate import interp1d

N = inp.pts_phiDM
data = np.ndarray((N,3))
y1 = []
y2 = []
for j in range(N):
    filename = '../../testData/Data_5_4/'+'testDM_phi='+str("{:4.4f}".format(inp.range_phi[j])).replace('.','')+'.npy'
    data[j] = np.load(filename)
    y1.append(data[j,1])
    y2.append(data[j,2])

x = inp.range_phi
y1 = np.array(y1)
y2 = np.array(y2)

func1 = interp1d(x,y1,'cubic')
func2 = interp1d(x,y2,'cubic')

plt.figure(figsize=(8,8))

plt.plot(x,y1,'k*')
X = np.linspace(0,x[-1],100)
plt.plot(X,func1(X),'r')

plt.plot(x,y2,'b^')
X = np.linspace(0,x[-1],100)
plt.plot(X,func2(X),'g')

plt.show()
