import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import tensorcircuit as tc
n=int(input())
X,Z,I=tc.gates._x_matrix,tc.gates._z_matrix,tc.gates._i_matrix
H=np.zeros((2**n,2**n))
for i in range(n):
    a=[[1]]
    for j in range(i):
        a=np.kron(a,I)
    a=np.kron(a,Z)
    for j in range(n-i-1):
        a=np.kron(a,I)
    H=H+a
for i in range(n-1):
    a=[[1]]
    for j in range(i):
        a=np.kron(a,I)
    a=np.kron(a,X)
    a=np.kron(a,X)
    for j in range(n-i-2):
        a=np.kron(a,I)
    H=H+a
print(H)
print(H[0,0])