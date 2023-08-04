import tensorcircuit as tc
import numpy as np
import tensorflow as tf
import logging
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

K = tc.set_backend("tensorflow")

n = 6
nlayers = 3
X,Z,I=tc.gates._x_matrix,tc.gates._z_matrix,tc.gates._i_matrix

def get_circuit(params):
    c = tc.Circuit(n)
    for i in range(n):
        c.h(i)
    for i in range(nlayers):
        for j in range(n - 1):
            c.exp1(j, j + 1, unitary=tc.gates._zz_matrix, theta=params[i * 2, j])
        for j in range(n):
            c.rx(j,theta=params[i * 2 + 1, j])
    return c

def get_H():
    H = K.zeros((2 ** n, 2 ** n))
    for i in range(n):
        a = [[1]]
        for j in range(i):
            a = np.kron(a, I)
        a = np.kron(a, X)
        for j in range(n - i - 1):
            a = np.kron(a, I)
        H = H - a
    for i in range(n - 1):
        a = [[1]]
        for j in range(i):
            a = np.kron(a, I)
        a = np.kron(a, Z)
        a = np.kron(a, Z)
        for j in range(n - i - 2):
            a = np.kron(a, I)
        H = H + a
    return H

H = get_H()

def get_loss(params):
    c = get_circuit(params)
    ex = c.expectation([H, range(n)])
    return K.convert_to_tensor(K.real(ex))

opt = K.optimizer(tf.keras.optimizers.Adam(1e-2))
loss_and_grad = K.jit(K.value_and_grad(get_loss, argnums=0))

def grad_optimize(step):
    params = tf.Variable(
        initial_value=tf.random.normal(
            shape=[nlayers * 2, n], stddev=0.1, dtype=getattr(tf, tc.rdtypestr)
        )
    )
    #params = np.random.uniform(low=-10.0, high=10.0, size=[2 * nlayers, n])
    #params = K.cast(K.convert_to_tensor(params), "float64")
    for i in range(step):
        loss, grad = loss_and_grad(params)
        params = opt.update(grad, params)
        if(i % 500 == 0):
            print(loss)

    loss, grad = loss_and_grad(params)
    print(loss)

grad_optimize(5000)
