"""Offline Exp D. (a) time compression by rescaling switch times then re-landing.
(b) search resolution and goal-relaxation sweep. No dev episode used."""
import multiprocessing as mp
import sys
import time

import numpy as np

import acrobot_model as m
import reach_search as rs
import switch_land as sl
import traj_opt as to

K_HOLD, P_HOLD = to.upright_lqr()
STARTS = [
    ("exp04", [1.7291055084321112, -0.36881622455681984, 0, 0]),
    ("ep1", [1.003622495443695, 2.0189631337802245, 0, 0]),
    ("hanging", [np.pi - 0.01, 0.01, 0, 0]),
    ("folded", [2.5, -2.5, 0, 0]),
    ("high", [0.6, -0.8, 0, 0]),
    ("low2", [-2.8, 1.0, 0, 0]),
]


def plan_return(x0, X, U):
    Ks = to.tvlqr(X, U, P_HOLD)
    res = to.closed_loop(x0, X, U, Ks, K_HOLD)
    r = res["rewards"]
    return float(res["ret"]), (int(np.argmax(r >= 1.0)) + 1 if (r >= 1.0).any() else None)


def seq_from_tau(s0, tau, T):
    return sl.switches_to_controls(s0, tau, T)


def land_from_tau(x0, s0, tau, T, target, iters):
    """LM restart from explicit switch times."""
    U = seq_from_tau(s0, tau, T)
    return sl.land_switches_tau(x0, s0, tau, T, P_HOLD, iters=iters, target=target)


def compress(x0, lm, target=2.0, iters=40, budget=120.0):
    t0 = time.time()
    s0, tau, T = lm["s0"], lm["tau"].copy(), len(lm["U"])
    best = lm
    steps = [8, 4, 2, 1]
    i = 0
    while i < len(steps) and time.time() - t0 < budget:
        dT = steps[i]
        Tn = T - dT
        taun = sl.project_order(tau * (Tn / T), Tn)
        r = sl.land_switches_tau(x0, s0, taun, Tn, P_HOLD, iters=iters, target=target)
        if r["V"] < target:
            tau, T, best = r["tau"], Tn, r
        else:
            i += 1
    return best


def job_compress(args):
    name, x0, score, layer, seq = args
    t0 = time.time()
    lm = sl.land_switches(x0, seq, P_HOLD, iters=120, target=2.0)
    if lm["V"] >= 2.0:
        return dict(name=name, layer=layer, ok=False, V=lm["V"], t=time.time() - t0)
    ret0, fb0 = plan_return(x0, lm["X"], lm["U"])
    c = compress(x0, lm)
    ret1, fb1 = plan_return(x0, c["X"], c["U"])
    return dict(name=name, layer=layer, ok=True, T0=len(lm["U"]), ret0=ret0, fb0=fb0, T1=len(c["U"]), ret1=ret1, fb1=fb1,
                nsw=len(c["tau"]), t=time.time() - t0)


def search_one(item):
    name, x0 = item
    cands, st = rs.search(np.array(x0, float), extra_layers=6, top_k=16)
    return name, np.array(x0, float), sorted(cands, key=lambda c: (c[1], c[0])), st


def job_sweep(args):
    name, x0, hold, shape, e_margin, goal = args
    t0 = time.time()
    cands, st = rs.search(np.array(x0, float), hold=hold, shape=shape, e_margin=e_margin, goal=goal,
                          extra_layers=0, top_k=4, max_layers=int(900 / hold), cap=1500000)
    fl = st["found_layer"]
    return dict(name=name, hold=hold, shape=shape, e_margin=e_margin, goal=goal, T_found=None if fl is None else fl * hold,
                peak=st["peak"], secs=round(time.time() - t0, 1))


if __name__ == "__main__":
    t0 = time.time()
    with mp.Pool(6) as pool:
        searched = pool.map(search_one, STARTS)
    jobs = []
    for name, x0, cands, st in searched:
        # diversify over layers rather than taking only the earliest ones
        idx = np.linspace(0, len(cands) - 1, min(16, len(cands))).astype(int)
        for i in idx:
            score, layer, seq = cands[i]
            jobs.append((name, x0, score, layer, seq))
    print("compress jobs", len(jobs), flush=True)
    with mp.Pool(48) as pool:
        results = pool.map(job_compress, jobs, chunksize=1)
    print("compress elapsed", round(time.time() - t0, 1))
    for name, *_ in STARTS:
        rows = [r for r in results if r["name"] == name and r["ok"]]
        if not rows:
            print(name, "no landed candidates")
            continue
        b0 = max(rows, key=lambda r: r["ret0"])
        b1 = max(rows, key=lambda r: r["ret1"])
        print(f"{name}: landed {len(rows)} | best before ret {b0['ret0']:.1f} T {b0['T0']} fb {b0['fb0']} | "
              f"best after ret {b1['ret1']:.1f} T {b1['T1']} fb {b1['fb1']} nsw {b1['nsw']} | "
              f"mean T saved {np.mean([r['T0'] - r['T1'] for r in rows]):.1f} | mean t {np.mean([r['t'] for r in rows]):.0f}s", flush=True)
    t1 = time.time()
    sweep = []
    for name, x0 in (STARTS[0], STARTS[2], STARTS[5]):
        for hold in (8, 5):
            for shape in ((64, 64, 40, 60), (128, 128, 80, 120)):
                for e_margin in (8.0, 3.0):
                    for goal in (1.0, 3.0):
                        sweep.append((name, x0, hold, shape, e_margin, goal))
    with mp.Pool(48) as pool:
        rows = pool.map(job_sweep, sweep, chunksize=1)
    print("sweep elapsed", round(time.time() - t1, 1))
    for r in rows:
        print(r)
