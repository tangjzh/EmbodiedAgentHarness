"""Offline stress: grid of starts on the perturbed plant + large-budget bound on catch time. No dev episodes."""
import time
import numpy as np
import controller_v2 as C
import eval_sim
import mysim

pert = mysim.Params(mc=mysim.MC * 1.05, mb=mysim.MB * 0.95, tc=0.025, mu=0.7, dmin=0.85)


def blocked(bx, bz):
    lx, lz = bx, (0.2 + bz) - 0.6
    for (x1, z1, x2, z2) in mysim.CAPS:
        sx, sz = x2 - x1, z2 - z1
        tt = max(0.0, min(1.0, ((lx - x1) * sx + (lz - z1) * sz) / (sx * sx + sz * sz)))
        if np.hypot(lx - (x1 + tt * sx), lz - (z1 + tt * sz)) < mysim.R_BALL + mysim.R_CAP:
            return True
    return False


t0 = time.time()
xs = np.linspace(-0.2, 0.2, 9)
zs = np.linspace(0.2, 0.5, 7)
print("Grid on PERTURBED plant: first reward tick (F = no sustained catch in 200 ticks, -- = start penetrates cup)")
print("  z\\x " + " ".join(f"{x:+.2f}" for x in xs))
firsts, fails = [], []
for z in zs[::-1]:
    row = []
    for x in xs:
        if blocked(x, z):
            row.append("  --")
            continue
        ret, first = eval_sim.run_episode(C, float(x), float(z), P=pert, T=200)
        ok = first is not None and ret >= 0.8 * (200 - first)
        if ok:
            firsts.append(first)
        else:
            fails.append((round(float(x), 3), round(float(z), 3), ret, first))
        row.append(f"{first:4d}" if ok else "   F")
    print(f"{z:.3f} " + " ".join(f"{c:>5}" for c in row), flush=True)
print("n ok", len(firsts), "fails", fails, "median first", int(np.median(firsts)), "max first", max(firsts), f"[{time.time()-t0:.0f}s]")

print("\nDoes more search shorten catches? (nominal plant, 5 planner seeds per setting, first-reward ticks)")
starts = {"exp2": (0.1101, 0.3324), "exp3": (-0.1554, 0.2799), "low": (0.05, 0.2)}
for name, par in [("base iters0=10 N=64", {}), ("iters0=40", {"iters0": 40}), ("iters0=40 N=128 iters=3", {"iters0": 40, "N": 128, "iters": 3})]:
    t1 = time.time()
    out = {}
    for sname, (bx, bz) in starts.items():
        fs = []
        for seed in range(5):
            p = dict(C.PAR, seed=seed, **par)
            ret, first = eval_sim.run_episode(C, bx, bz, T=120, par=p)
            fs.append(first if first is not None else 999)
        out[sname] = fs
    print(f"{name:26s}", out, f"[{time.time()-t1:.0f}s]", flush=True)
