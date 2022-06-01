import matplotlib.pyplot as plt
import numpy as np
import initial_states as ins

ru = np.pi/4
rx = np.pi/7
rz1 = 2*np.pi/6
rz2 = 3*np.pi/2
a = np.array([np.cos(ru),0,-np.sin(ru)])
b = np.array([np.cos(ru),np.sin(ru)*np.sqrt(3)/2,np.sin(ru)/2])
c = np.array([np.cos(ru),-np.sin(ru)*np.sqrt(3)/2,np.sin(ru)/2])
#
a1 = np.tensordot(ins.R_x(rx),a,1)
b1 = np.tensordot(ins.R_x(rx),b,1)
c1 = np.tensordot(ins.R_x(rx),c,1)
#
b2 = np.tensordot(ins.R_z(rz1),b1,1)
a2 = a1
c2 = np.tensordot(ins.R_z(rz2),c1,1)
P = [ru,rx,rz1,rz2,True]
ins.full_func(P)

exit()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

V = [[a2,'r'],[b2,'g'],[c2,'b']]
P = [[0,0,0,1/4,np.sqrt(3)/2,-np.sqrt(3)/4,'b'],[0,0,0,np.sqrt(3)/2,1/2,0,'g'],[0,0,0,np.sqrt(3)/2,-1/4,np.sqrt(3)/4,'r']]

for p in V:
    ax.quiver(0,0,0,p[0][0],p[0][1],p[0][2],color = p[1])
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
plt.show()
