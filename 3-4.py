import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import tensorcircuit as tc
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
K = tc.set_backend("tensorflow")

dtype_complex="complex128"
dtype_int="int64"
dtype_float="float32"
a=np.array([[[0,1],[1,0]],[[0,0-1j],[0+1j,0]],[[1,0],[0,-1]]])
tp1=0
tp2=1
p,q=a[tp1],a[tp2]
def ex(theta:tf.Tensor)->tf.Tensor:
    v = tf.linalg.expm(-0.5j * theta * p) @ q @ tf.linalg.expm(0.5j * theta * p)
    return v[0, 0]
der=K.grad(ex)
N=20
t=np.linspace(-np.pi,np.pi,N)
ep=[]
for i in range(N):
    ep.append(der(K.cast(K.convert_to_tensor(t[i]),dtype_complex)))
ep=K.real(ep)
plt.plot(t,ep,'-');
plt.show()