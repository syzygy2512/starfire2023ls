import tensorcircuit as tc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

K = tc.set_backend("tensorflow")
dtype_float = "float64"

nsamples = 100
xs0 = np.random.uniform(low=-10.0, high=10.0, size=[nsamples])
ys0 = xs0**2 + np.sin(xs0)*30
#ys0 = np.random.uniform(low=-3.0, high=0.0, size=[nsamples])
ans = [0.0, 0.0]
xs0 = K.convert_to_tensor(xs0)
ys0 = K.convert_to_tensor(ys0)
ans = K.cast(K.convert_to_tensor(ans), dtype_float)

@K.jit
def dis(x, y, params):
    return K.sqrt((x - params[0]) ** 2 + (y - params[1]) ** 2)

dis_vmap = K.jit(K.vmap(dis, vectorized_argnums=(0, 1)))

@K.jit
def dis_all(xs, ys, params):
    return K.sum(dis_vmap(xs, ys, params))

dis_and_grad = K.jit(K.value_and_grad(dis_all, argnums=2))
opt = K.optimizer(tf.keras.optimizers.Adam(1e-2))
maxstep = 5000

for i in range(maxstep):
    ds, grad = dis_and_grad(xs0, ys0, ans)
    ans = opt.update(grad, ans)
    if i % 1000 == 0 :
        print(ds)
print(ans)
plt.plot(xs0, ys0, 'o')
plt.plot(ans[0], ans[1], 'o')