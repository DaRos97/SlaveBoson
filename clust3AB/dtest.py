import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

x = np.array([0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2])
y1 = np.array([ -0.46821,       #3x3
                -0.51127,
                -0.56158,
                -0.60393,
                -0.63283,
                -0.64558,
                -0.64114,
                -0.61993,
                -0.58387,
                -0.53699,
                -0.48774,
                -0.49322
    ])
y2 = np.array([ -0.46742,        #q0
                -0.46944,
                -0.47461,
                -0.48087,
                -0.48598,
                -0.48845,
                -0.48765,
                -0.48376,
                -0.47757,
                -0.47080,
                -0.46590,
                -0.46510
    ])
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
