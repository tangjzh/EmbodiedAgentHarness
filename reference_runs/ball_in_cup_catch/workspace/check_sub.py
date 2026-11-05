"""Real accuracy check: does the planner's batch model predict logged REAL motion, and is sub=5 as good as sub=10?
Uses the same physics as controller_v2.rollout via a probe copy: forward 25 ticks open-loop from logged states."""
import inspect
import re
import numpy as np
import controller_v2 as C

src = inspect.getsource(C.rollout)
# Build a probe that also returns the final state; the only edit is the return statement and the hold override.
src = src.replace("def rollout(", "def probe(").replace("return score, cnt", "return np.stack([cx, cz, bx, bz], 1)")
src = re.sub(r"ux = np\.where\(cup,.*?U\[:, t, 0\]\)", "ux = U[:, t, 0]", src, flags=re.S)
src = re.sub(r"uz = np\.where\(cup,.*?U\[:, t, 1\]\)", "uz = U[:, t, 1]", src, flags=re.S)
ns = dict(vars(C))
exec(src, ns)
probe = ns["probe"]

L = np.load("memory/exp01.npz")["log"]
U, Q, V = L[:, :2], L[:, 2:6], L[:, 6:10]
lz = (0.2 + Q[:, 3]) - (0.6 + Q[:, 1])
lx = Q[:, 2] - Q[:, 0]
for horizon in (5, 25, 50):
    for sub in (10, 5):
        par = dict(C.PAR, sub=sub)
        e_free, e_con = [], []
        for i in range(0, len(L) - horizon - 1, 5):
            s = tuple(Q[i]) + tuple(V[i])
            seg = U[i + 1:i + 1 + horizon][None]
            end = probe(s, seg, par)[0]
            err = np.abs(end - Q[i + horizon]).max()
            window = range(i, i + horizon + 1)
            contact = any(abs(lx[j]) < 0.1 and -0.15 < lz[j] < 0.05 for j in window)
            (e_con if contact else e_free).append(err)
        fm = np.median(e_free) if e_free else float("nan")
        fp = np.percentile(e_free, 95) if e_free else float("nan")
        cm = np.median(e_con) if e_con else float("nan")
        cp = np.percentile(e_con, 95) if e_con else float("nan")
        print(f"horizon {horizon:2d} ticks sub {sub:2d}: no-contact n={len(e_free)} median {fm:.4f} p95 {fp:.4f} m | "
              f"near-cup n={len(e_con)} median {cm:.4f} p95 {cp:.4f} m")
