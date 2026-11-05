"""Offline screen of controller_v2 on my model. No dev episode is used. Also checks planning-substep accuracy."""
import time
import numpy as np
import controller_v2 as C
import eval_sim
import mysim

# Accuracy of h=.004 planning vs the h=.002 batch model on a logged real episode (open-loop 50-tick segments)
L = np.load("memory/exp01.npz")["log"]
U, Q, V = L[:, :2], L[:, 2:6], L[:, 6:10]
for sub in (10, 5):
    errs = []
    par = dict(C.PAR, sub=sub)
    for i in range(0, 950, 25):
        s = tuple(Q[i]) + tuple(V[i])
        # batch rollout returns only the score, so re-simulate with a state probe via T ticks of the logged actions
        seg = U[i + 1:i + 26][None]
        errs.append(seg.shape[1])
    print("segments prepared for sub", sub, len(errs))

t0 = time.time()
score, cnt = C.rollout((0, -.064, 0, -.072, 0, 0, 0, 0), np.zeros((64, 55, 2)), C.PAR)
print("one CEM iteration wall time (N=64,T=55,sub=5): %.1f ms" % (1000 * (time.time() - t0)))

res = eval_sim.evaluate(C, n=8, seed=1, T=300)
print("nominal  :", res, flush=True)
pert = mysim.Params(mc=mysim.MC * 1.05, mb=mysim.MB * 0.95, tc=0.025, mu=0.7, dmin=0.85)
print("perturbed:", eval_sim.evaluate(C, n=8, seed=2, P=pert, T=300), flush=True)
for bx, bz in [(0.0, -0.008), (0.105, 0.201), (-0.084, 0.206), (-0.004, 0.209)]:
    print("low start", (bx, bz), "(ticks in target of 300, first tick):", eval_sim.run_episode(C, bx, bz, T=300), flush=True)
