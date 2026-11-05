"""Offline Exp G: does candidate diversity or resolution buy return after landing?

Config A: one low-res search, as in v1.
Config B: several low-res searches with random cell phase, hold and goal.
Config C: high-res searches.
For every config: best closed-loop return after LM, and after iLQR polish of the best few.
"""
import multiprocessing as mp
import time

import numpy as np

import acrobot_model as m
import reach_search as rs
import switch_land as sl
import traj_opt as to

K_HOLD, P_HOLD = to.upright_lqr()
LOW = (64, 64, 40, 60)
HIGH = (128, 128, 80, 120)
STARTS = [
    ("exp04", [1.7291055084321112, -0.36881622455681984]),
    ("ep1", [1.003622495443695, 2.0189631337802245]),
    ("hanging", [np.pi - 0.01, 0.01]),
    ("folded", [2.5, -2.5]),
    ("high", [0.6, -0.8]),
    ("low2", [-2.8, 1.0]),
]


def closed(x0, X, U):
    Ks = to.tvlqr(X, U, P_HOLD)
    res = to.closed_loop(x0, X, U, Ks, K_HOLD)
    r = res["rewards"]
    return float(res["ret"]), (int(np.argmax(r >= 1.0)) + 1 if (r >= 1).any() else None), bool(r[-1] >= 1.0)


def job_search(args):
    name, q, config, kw = args
    x0 = np.array([q[0], q[1], 0.0, 0.0])
    t0 = time.time()
    cands, st = rs.search(x0, **kw)
    cands = sorted(cands, key=lambda c: (c[1], c[0]))[:32]
    return dict(name=name, config=config, x0=x0, cands=cands, hold=kw.get("hold", 8), T_found=None if st["found_layer"] is None else st["found_layer"] * kw.get("hold", 8), secs=time.time() - t0)


def job_land(args):
    name, config, x0, score, layer, seq = args
    t0 = time.time()
    lm = sl.land_switches(x0, seq, P_HOLD, iters=60, target=5.0)
    ret, first, held = closed(x0, lm["X"], lm["U"])
    return dict(name=name, config=config, T=len(seq), V=float(lm["V"]), ret=ret, first=first, held=held, U=lm["U"], secs=time.time() - t0)


def job_polish(args):
    name, config, x0, U = args
    t0 = time.time()
    X = to.rollout_open(x0, U)
    x = X[-1].copy()
    hold = np.zeros(150)
    for t in range(150):
        xw = x.copy()
        xw[:2] = to.wrap(xw[:2])
        hold[t] = float(np.clip(-K_HOLD @ xw, -1, 1))
        x = m.step(x, hold[t])
    U0 = np.concatenate([U, hold])
    s = to.ilqr(x0, U0, to.Cost(len(U0), 1.0, 0.0, 0.0, 1e-5, P_HOLD * 1e-2), iters=60)
    ret, first, held = closed(x0, s["X"], s["U"])
    return dict(name=name, config=config, ret=ret, first=first, held=held, secs=time.time() - t0)


if __name__ == "__main__":
    t_all = time.time()
    rng = np.random.default_rng(5)
    jobs = []
    for name, q in STARTS:
        jobs.append((name, q, "A", dict(extra_layers=6, top_k=16)))
        for i in range(12):
            jobs.append((name, q, "B", dict(extra_layers=4, top_k=8, hold=int(rng.choice([6, 7, 8, 9, 10])), goal=float(rng.choice([1.0, 2.0, 3.0])),
                                            phase=tuple(rng.uniform(0, 1, 4)), max_layers=int(900 / 6))))
        for hold in (5, 8):
            for goal in (1.0, 3.0):
                jobs.append((name, q, "C", dict(extra_layers=4, top_k=8, hold=hold, goal=goal, shape=HIGH, max_layers=int(900 / hold))))
    with mp.Pool(64) as pool:
        searched = pool.map(job_search, jobs, chunksize=1)
        land_jobs = []
        for s in searched:
            for score, layer, seq in s["cands"]:
                land_jobs.append((s["name"], s["config"], s["x0"], score, layer, seq))
        print("search secs by config:", {c: round(float(np.mean([s["secs"] for s in searched if s["config"] == c])), 1) for c in "ABC"}, "land jobs", len(land_jobs), "elapsed", round(time.time() - t_all, 1), flush=True)
        landed = pool.map(job_land, land_jobs, chunksize=4)
        print("mean LM secs", round(float(np.mean([l["secs"] for l in landed])), 2), "elapsed", round(time.time() - t_all, 1), flush=True)
        polish_jobs = []
        x0s = {s["name"]: s["x0"] for s in searched}
        for name, _ in STARTS:
            for config in "ABC":
                rows = sorted([l for l in landed if l["name"] == name and l["config"] == config], key=lambda l: -l["ret"])[:4]
                for l in rows:
                    polish_jobs.append((name, config, x0s[name], l["U"]))
        polished = pool.map(job_polish, polish_jobs, chunksize=1)
    print("elapsed", round(time.time() - t_all, 1))
    totals = {c: [] for c in "ABC"}
    for name, _ in STARTS:
        print(name)
        for config in "ABC":
            rows = [l for l in landed if l["name"] == name and l["config"] == config]
            held = [l for l in rows if l["held"]]
            best = max(rows, key=lambda l: l["ret"])
            pols = [p for p in polished if p["name"] == name and p["config"] == config]
            best_pol = max(pols, key=lambda p: p["ret"])
            overall = max(best["ret"], best_pol["ret"])
            totals[config].append(overall)
            Tf = [s["T_found"] for s in searched if s["name"] == name and s["config"] == config and s["T_found"]]
            print(f"  {config}: n {len(rows)} held {len(held)} T_found_min {min(Tf)} | LM best {best['ret']:.1f} (T {best['T']}, first {best['first']}) | polish best {best_pol['ret']:.1f} (first {best_pol['first']}, {best_pol['secs']:.0f}s)")
    print({c: round(float(np.mean(v)), 1) for c, v in totals.items()})
