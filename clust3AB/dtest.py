import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

x = np.array([0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2])
y1 = np.array([ -0.46821,       #3x3
                -0.51127,
                -0.56158,
                -0.,
                -0.,
                -0.,
                -0.,
                -0.,
                -0.,
                -0.,
                -0.,
                -0.,
    ])
y2 = np.array([ -0.46742,        #q0
                -0.46944,
                -0.47461,
                -0.,
                -0.,
                -0.,
                -0.,
                -0.,
                -0.,
                -0.,
                -0.,
                -0.,
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
