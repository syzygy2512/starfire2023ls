import numpy as np
import tensorcircuit as tc
c = tc.Circuit(2)
c.h(0)
c.cx(0, 1)
N=20
for i in range(N):
    a=c.measure(0,1)[0]
    print(a)