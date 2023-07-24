import tensorcircuit as tc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

dtype_float="float32"

nsamples = 100
k0 = 2
b0 = 1
xs0 = np.random.uniform(low=-1.0, high=1.0, size=[nsamples])
ys0 = k0 * xs0 + b0 + np.random.normal(scale=0.1, size=[nsamples])
xs0 = tc.backend.convert_to_tensor(xs0)
ys0 = tc.backend.convert_to_tensor(ys0)

def loss(x,y,params):
    yy = params[0] * x + params[1]
    return (yy - y) ** 2

loss_vmap = tc.backend.vmap(loss,vectorized_argnums=(0, 1))

def loss_all(xs,ys,params):
    return tc.backend.sum(loss_vmap(xs, ys, params))

loss_and_grad = tc.backend.jit(tc.backend.value_and_grad(loss_all, argnums=2))
maxstep = 500
st = [0,0]
st = tc.backend.cast(tc.backend.convert_to_tensor(st),"float64")
opt = tc.backend.optimizer(tf.keras.optimizers.Adam(1e-2))

for i in range(maxstep):
    ls, grad = loss_and_grad(xs0, ys0, st)
    st = opt.update(grad, st)
    if(i % 100 == 0):
        print(ls)
print(st)
plt.plot(xs0, ys0, 'o')
xl = np.linspace(-1, 1, 100)
yl = st[0] * xl + st[1]
plt.plot(xl, yl, '-')
plt.show()