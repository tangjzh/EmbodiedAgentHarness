"""Offline Exp F: why do specific starts fail, and does a minimum arrival time fix near-upright ones?"""
import multiprocessing as mp
import time

import numpy as np

import acrobot_model as m
import reach_search as rs
import switch_land as sl
import traj_opt as to

K_HOLD, P_HOLD = to.upright_lqr()
FAILS = [("fail_a", [-0.607, 2.789]), ("up_a", [-0.097, 0.145]), ("up_b", [0.02, 0.02]), ("up_c", [0.1, -0.1]), ("up_d", [0.05, 0.0]), ("up_e", [-0.01, 0.03])]


def closed(x0, X, U):
    Ks = to.tvlqr(X, U, P_HOLD)
    res = to.closed_loop(x0, X, U, Ks, K_HOLD)
    r = res["rewards"]
    return float(res["ret"]), (int(np.argmax(r >= 1.0)) + 1 if (r >= 1).any() else None), bool(r[-1] >= 1.0)


def run(args):
    name, q, min_T, target, iters = args
    x0 = np.array([q[0], q[1], 0.0, 0.0])
    t0 = time.time()
    cands, st = rs.search(x0, extra_layers=6, top_k=16, min_layer=int(min_T / 8))
    t_search = time.time() - t0
    cands = sorted(cands, key=lambda c: (c[1], c[0]))
    rows = []
    for score, layer, seq in cands[:10]:
        t1 = time.time()
        lm = sl.land_switches(x0, seq, P_HOLD, iters=iters, target=target)
        rows.append((layer * 8, round(float(lm["V"]), 2), lm["iters"], round(closed(x0, lm["X"], lm["U"])[0], 1), round(time.time() - t1, 1)))
    return dict(name=name, min_T=min_T, target=target, iters=iters, found=st["found_layer"], layers=st["layers"], peak=st["peak"], n_cands=len(cands),
                t_search=round(t_search, 1), rows=rows)


if __name__ == "__main__":
    jobs = []
    for name, q in FAILS:
        for min_T in (0, 80, 160, 240):
            jobs.append((name, q, min_T, 5.0, 60))
    with mp.Pool(32) as pool:
        out = pool.map(run, jobs, chunksize=1)
    for o in out:
        best = max((r[3] for r in o["rows"]), default=None)
        print(o["name"], "min_T", o["min_T"], "found", o["found"], "layers", o["layers"], "n_cands", o["n_cands"], "t_search", o["t_search"], "BEST", best)
        print("    (T, V, iters, ret, secs):", o["rows"][:6])
