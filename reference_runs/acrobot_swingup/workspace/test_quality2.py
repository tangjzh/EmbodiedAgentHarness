"""Offline Exp G (retry): candidate diversity vs resolution, with a deadline and a log file.

CONFIG file selects which configs to run: "AB" (cheap) or "C" (high-res).
A: one low-res search as in v1. B: 12 low-res searches with random phase/hold/goal.
C: high-res searches.
"""
import os

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"

import json
import multiprocessing as mp
import sys
import time

import numpy as np

import acrobot_model as m
import reach_search as rs
import switch_land as sl
import traj_opt as to

K_HOLD, P_HOLD = to.upright_lqr()
HIGH = (128, 128, 80, 120)
STARTS = [
    ("exp04", [1.7291055084321112, -0.36881622455681984]),
    ("ep1", [1.003622495443695, 2.0189631337802245]),
    ("hanging", [np.pi - 0.01, 0.01]),
    ("folded", [2.5, -2.5]),
    ("high", [0.6, -0.8]),
    ("low2", [-2.8, 1.0]),
]
DEADLINE_S = 500.0


def closed(x0, X, U):
    Ks = to.tvlqr(X, U, P_HOLD)
    res = to.closed_loop(x0, X, U, Ks, K_HOLD)
    r = res["rewards"]
    return float(res["ret"]), (int(np.argmax(r >= 1.0)) + 1 if (r >= 1).any() else None), bool(r[-1] >= 1.0)


def job(args):
    """One search, landing its candidates. Returns compact per-search results."""
    name, q, config, kw, n_land = args
    x0 = np.array([q[0], q[1], 0.0, 0.0])
    t0 = time.time()
    cands, st = rs.search(x0, **kw)
    t_search = time.time() - t0
    cands = sorted(cands, key=lambda c: (c[1], c[0]))[:n_land]
    best = None
    rets = []
    for score, layer, seq in cands:
        lm = sl.land_switches(x0, seq, P_HOLD, iters=60, target=5.0)
        ret, first, held = closed(x0, lm["X"], lm["U"])
        rets.append(ret)
        if best is None or ret > best[0]:
            best = (ret, first, held, len(seq))
    hold = kw.get("hold", 8)
    fl = st["found_layer"]
    return dict(name=name, config=config, hold=hold, goal=kw.get("goal", 1.0), T_found=None if fl is None else fl * hold,
                t_search=round(t_search, 1), t_total=round(time.time() - t0, 1), n=len(cands), best=best,
                median_ret=float(np.median(rets)) if rets else None)


if __name__ == "__main__":
    which = open("QUALITY_CONFIG.txt").read().strip()
    t_all = time.time()
    rng = np.random.default_rng(5)
    jobs = []
    for name, q in STARTS:
        if "A" in which:
            jobs.append((name, q, "A", dict(extra_layers=6, top_k=16), 32))
        if "B" in which:
            for i in range(12):
                jobs.append((name, q, "B", dict(extra_layers=4, top_k=8, hold=int(rng.choice([6, 7, 8, 9, 10])), goal=float(rng.choice([1.0, 2.0, 3.0])),
                                                phase=tuple(rng.uniform(0, 1, 4)), max_layers=150), 12))
        if "C" in which:
            for hold, goal in ((8, 1.0), (8, 3.0), (5, 3.0)):
                jobs.append((name, q, "C", dict(extra_layers=4, top_k=8, hold=hold, goal=goal, shape=HIGH, max_layers=int(900 / hold)), 12))
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/quality_{which}.jsonl"
    rows = []
    done = 0
    with open(log_path, "w") as log, mp.Pool(min(64, len(jobs))) as pool:
        it = pool.imap_unordered(job, jobs, chunksize=1)
        while done < len(jobs):
            remaining = DEADLINE_S - (time.time() - t_all)
            if remaining <= 0:
                print("DEADLINE hit, partial results", done, "of", len(jobs))
                pool.terminate()
                break
            try:
                r = it.next(timeout=remaining)
            except mp.TimeoutError:
                print("DEADLINE hit, partial results", done, "of", len(jobs))
                pool.terminate()
                break
            rows.append(r)
            done += 1
            log.write(json.dumps(r, default=float) + "\n")
            log.flush()
    print("elapsed", round(time.time() - t_all, 1), "jobs done", done, "of", len(jobs))
    totals = {}
    for name, _ in STARTS:
        print(name)
        for config in which:
            sub = [r for r in rows if r["name"] == name and r["config"] == config and r["best"]]
            if not sub:
                print("  ", config, "no results")
                continue
            b = max(sub, key=lambda r: r["best"][0])
            totals.setdefault(config, []).append(b["best"][0])
            single = [r["best"][0] for r in sub]
            print(f"  {config}: searches {len(sub)} T_found_min {min(r['T_found'] for r in sub if r['T_found'])} "
                  f"best ret {b['best'][0]:.1f} first {b['best'][1]} held {b['best'][2]} (hold {b['hold']} goal {b['goal']}) "
                  f"median-of-search-bests {np.median(single):.1f} mean_t_search {np.mean([r['t_search'] for r in sub]):.1f}s mean_t_total {np.mean([r['t_total'] for r in sub]):.1f}s")
    print({c: round(float(np.mean(v)), 1) for c, v in totals.items()})
