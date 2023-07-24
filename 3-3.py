import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
a=np.array([[[0,1],[1,0]],[[0,0-1j],[0+1j,0]],[[1,0],[0,-1]]])
tp1=0
tp2=1
p,q=a[tp1],a[tp2]
N=200
t=np.linspace(-np.pi,np.pi,N)
ep=[0]*N
for i in range(N):
    v=sp.linalg.expm(-0.5j*t[i]*p)@q@sp.linalg.expm(0.5j*t[i]*p)
    ep[i]=v[0,0]
plt.plot(t,ep,'-');
plt.show()
