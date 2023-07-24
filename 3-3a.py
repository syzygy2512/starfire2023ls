import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
a=np.array([[[0,1],[1,0]],[[0,0-1j],[0+1j,0]],[[1,0],[0,-1]]])
tp1=0
tp2=1
p,q=a[tp1],a[tp2]
delta=0.01
def ex(theta):
    v = sp.linalg.expm(-0.5j * theta * p) @ q @ sp.linalg.expm(0.5j * theta * p)
    return v[0, 0]
def der(theta):
    return (ex(theta+delta)-ex(theta))/delta
N=200
t=np.linspace(-np.pi,np.pi,N)
ep=[]
for i in range(N):
    ep.append(der(t[i]))
plt.plot(t,ep,'-');
plt.show()
