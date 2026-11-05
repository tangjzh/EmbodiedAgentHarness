"""Offline resource check. Does not import robot_api, so no episode budget is touched."""
import importlib
import multiprocessing as mp
import os
import shutil
import time

import numpy as np
import scipy

import acrobot_model as m

print("numpy", np.__version__, "scipy", scipy.__version__)
print("cpu_count", os.cpu_count(), "affinity", len(os.sched_getaffinity(0)))
for name in ("numba", "torch", "jax", "cython"):
    try:
        importlib.import_module(name)
        print(name, "available")
    except Exception as exc:
        print(name, "missing:", type(exc).__name__)
print("gcc", shutil.which("gcc"), "cc", shutil.which("cc"), "clang", shutil.which("clang"))

# batched rollout throughput
N = 2000
x = np.zeros((N, 4))
x[:, 0] = np.linspace(-3, 3, N)
u = np.ones(N)
t0 = time.time()
for _ in range(200):
    x = m.step(x, u)
dt = time.time() - t0
print(f"batched: {N * 200 / dt:.3g} state-steps/s")


def work(i):
    x = np.zeros((N, 4))
    x[:, 0] = np.linspace(-3, 3, N)
    for _ in range(200):
        x = m.step(x, np.ones(N))
    return i


t0 = time.time()
workers = min(8, len(os.sched_getaffinity(0)))
with mp.Pool(workers) as pool:
    pool.map(work, range(workers))
dt = time.time() - t0
print(f"multiprocessing with {workers} workers ok: {workers * N * 200 / dt:.3g} state-steps/s total")
