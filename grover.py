import tensorcircuit as tc
import numpy as np
"""
import cotengra

opt = cotengra.ReusableHyperOptimizer(
    methods=["greedy", "kahypar"],
    parallel="ray",
    minimize="combo",
    max_time=12,
    max_repeats=1024,
    progbar=True,
)
tc.set_contractor("custom", optimizer=opt, contraction_info=True, preprocessing=True)
"""
K = tc.set_backend("tensorflow")

N = 14
M = 6
U = N - 1
step = int(np.floor(np.pi / 4 * np.sqrt(2 ** 6 / 2)))
measure_time = 1000
c = tc.Circuit(N)

def qdiff(x, y, z):
    c.cnot(x, y)
    c.x(y)
    c.cnot(y, z)
    c.x(y)
    c.cnot(x, y)
    c.x(z)

def qdiff_all():
    qdiff(0, 1, 6)
    qdiff(2, 3, 7)
    qdiff(4, 5, 8)
    qdiff(0, 2, 9)
    qdiff(1, 3, 10)
    qdiff(2, 4, 11)
    qdiff(3, 5, 12)

def qmultiand():
    c.multicontrol(*range(M, U + 1), unitary=tc.gates.x(), ctrl=[1 for _ in range(M, U)])

def init():
    for i in range(M):
        c.h(i)
    c.x(U)
    c.h(U)

def oracle():
    qdiff_all()
    qmultiand()
    qdiff_all()

def grover_diffusion():
    for i in range(M):
        c.h(i)
        c.x(i)
    c.multicontrol(*range(M), unitary=tc.gates.z(), ctrl=[1 for _ in range(M - 1)])
    for i in range(M):
        c.x(i)
        c.h(i)

init()
for i in range(step):
    oracle()
    grover_diffusion()
for i in range(measure_time):
    measure_ans = c.measure(*range(6), with_prob=False)
    print(measure_ans[0])