import os
import sys
import time

if len(sys.argv) > 1 and sys.argv[1] == '1':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
t0 = time.time()
import numpy as np
print('import numpy', round(time.time() - t0, 2))
a = np.random.rand(409600, 3)
m = np.eye(3)
t0 = time.time()
b = a @ m.T
print('matmul 409600x3', round(time.time() - t0, 3))
t0 = time.time()
b2 = a[:, 0:1] * m[:, 0] + a[:, 1:2] * m[:, 1] + a[:, 2:3] * m[:, 2]
print('manual', round(time.time() - t0, 3), np.abs(b - b2).max())
t0 = time.time()
b3 = np.einsum('ij,kj->ik', a, m)
print('einsum', round(time.time() - t0, 3))
t0 = time.time()
c = np.cov(a[:20000, :2].T)
w, v = np.linalg.eigh(c)
print('cov+eigh', round(time.time() - t0, 3))
