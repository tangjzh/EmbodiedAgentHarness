"""Offline screening: run controller.policy on my planar model only (no robot.step, no dev episode used).

Ball starts are drawn x in [-.2,.2], z in [.2,.5], rejecting penetrations; these ranges are an assumption to be checked
against the starts actually observed on reset. A perturbed plant (mass/friction/constraint softness) reduces the inverse
crime of testing the controller's internal model against itself.
"""
import importlib
import math
import random
import sys
import time

import mysim


def sample_start(rng):
    while True:
        bx, bz = rng.uniform(-.2, .2), rng.uniform(.2, .5)
        lx, lz = bx, (0.2 + bz) - 0.6
        hit = False
        for (x1, z1, x2, z2) in mysim.CAPS:
            sx, sz = x2 - x1, z2 - z1
            tt = max(0.0, min(1.0, ((lx - x1) * sx + (lz - z1) * sz) / (sx * sx + sz * sz)))
            if math.hypot(lx - (x1 + tt * sx), lz - (z1 + tt * sz)) < mysim.R_BALL + mysim.R_CAP:
                hit = True
        if not hit:
            return bx, bz


def run_episode(ctrl, bx, bz, P=None, T=500, par=None, trace=False):
    sim = mysim.Sim(bx, bz, P=P)
    mem = {}
    if par is not None:
        mem["par"] = par
    ret, first, modes = 0, None, []
    obs = {"position": list(sim.q), "velocity": list(sim.v)}
    for k in range(T):
        a = ctrl.policy(obs, k, mem)
        r = sim.step(a)
        obs = {"position": list(sim.q), "velocity": list(sim.v)}
        if r:
            ret += 1
            if first is None:
                first = k
        if trace:
            modes.append((mem.get("mode"), list(a), list(sim.q)))
    return (ret, first, modes) if trace else (ret, first)


def evaluate(ctrl, n=40, seed=1, P=None, T=500, par=None):
    rng = random.Random(seed)
    rets, firsts, fails = [], [], []
    t0 = time.time()
    for i in range(n):
        bx, bz = sample_start(rng)
        ret, first = run_episode(ctrl, bx, bz, P=P, T=T, par=par)
        rets.append(ret / T)
        if first is None or ret < 0.5 * (T - first):
            fails.append((round(bx, 3), round(bz, 3), ret, first))
        if first is not None:
            firsts.append(first)
    firsts.sort()
    med = firsts[len(firsts) // 2] if firsts else None
    return {"mean_frac": round(sum(rets) / n, 3), "n_fail": len(fails), "median_first": med,
            "max_first": firsts[-1] if firsts else None, "sec": round(time.time() - t0, 1), "fails": fails[:6]}


if __name__ == "__main__":
    mod = sys.argv[2] if len(sys.argv) > 2 else "controller"
    ctrl = importlib.import_module(mod)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print("module:", mod, "n:", n)
    print("nominal  :", evaluate(ctrl, n=n, seed=1), flush=True)
    pert = mysim.Params(mc=mysim.MC * 1.05, mb=mysim.MB * 0.95, tc=0.025, mu=0.7, dmin=0.85)
    print("perturbed:", evaluate(ctrl, n=n, seed=2, P=pert), flush=True)
    print("settled-hang start (return ticks, first catch tick):", run_episode(ctrl, 0.0, -0.008, T=500), flush=True)
