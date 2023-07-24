import tensorcircuit as tc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

K = tc.set_backend("tensorflow")
dtype_float = "float64"

a = np.random.uniform(low=-1.0, high=1.0, size=[4])
a = K.convert_to_tensor(a)

@K.jit
def f(x):
    return a[0] * x ** 3 + a[1] * x ** 2 + a[2] * x + a[3]

@K.jit
def loss(x):
    return f(x) ** 2

loss_and_grad = K.jit(K.value_and_grad(loss))
opt = K.optimizer(tf.keras.optimizers.Adam(1e-2))
maxstep = 500
st = 0.0
st = K.cast(K.convert_to_tensor(st), dtype_float)

for i in range(maxstep):
    loss, grad = loss_and_grad(st)
    st = opt.update(grad, st)
    if i % 100 == 0 :
        print(loss)
print(st)

xl = np.linspace(-1,1,100)
yl = f(xl)

plt.plot(xl, yl, '-')
plt.plot(st, 0, 'o')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.show()
