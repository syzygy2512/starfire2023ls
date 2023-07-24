import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
a=np.array([[[0,1],[1,0]],[[0,0-1j],[0+1j,0]],[[1,0],[0,-1]]])
tp1=2
tp2=0
p,q=a[tp1],a[tp2]
N=200
t=np.linspace(-np.pi,np.pi,N)
v0=[1,0]
ep=[0]*N
for i in range(N):
    v=sp.linalg.expm(0.5j*t[i]*p)@v0
    ep[i]=v.conj().T@q@v
plt.plot(t,ep,'-');
plt.show()
