"""Offline Exp 3: switch-time LM landing, then iLQR polish on the native reward."""
import multiprocessing as mp
import time

import numpy as np

import acrobot_model as m
import reach_search as rs
import switch_land as sl
import traj_opt as to

K_HOLD, P_HOLD = to.upright_lqr()


def extend_with_hold(x_end, extra):
    """Simulate the saturating LQR forward so the plan includes a consistent hold segment."""
    x = x_end.copy()
    us = np.zeros(extra)
    for t in range(extra):
        xw = x.copy()
        xw[:2] = to.wrap(xw[:2])
        us[t] = float(np.clip(-K_HOLD @ xw, -1, 1))
        x = m.step(x, us[t])
    return us


def score_plan(x0, X, U):
    Ks = to.tvlqr(X, U, P_HOLD)
    res = to.closed_loop(x0, X, U, Ks, K_HOLD)
    r = res["rewards"]
    return res["ret"], (int(np.argmax(r >= 1.0)) + 1 if (r >= 1.0).any() else None), bool(r[-1] >= 1.0)


def attempt(args):
    name, x0, score, layer, seq = args
    t0 = time.time()
    lm = sl.land_switches(x0, seq, P_HOLD, iters=120, target=2.0)
    t_lm = time.time() - t0
    ret_lm, fb_lm, held_lm = score_plan(x0, lm["X"], lm["U"])
    out = dict(name=name, layer=layer, score=round(score, 3), V=float(lm["V"]), lm_iters=lm["iters"], ret_lm=round(ret_lm, 1),
               held_lm=held_lm, fb_lm=fb_lm, t_lm=round(t_lm, 1), ret_polish=None, fb_polish=None, t_polish=None)
    if lm["V"] < 2000.0:
        t1 = time.time()
        extra = 150
        U0 = np.concatenate([lm["U"], extend_with_hold(lm["X"][-1], extra)])
        T = len(U0)
        cost = to.Cost(T, 1.0, 0.0, 0.0, 1e-5, P_HOLD * 1e-2)
        s = to.ilqr(x0, U0, cost, iters=80)
        ret_p, fb_p, held_p = score_plan(x0, s["X"], s["U"])
        out.update(ret_polish=round(ret_p, 1), fb_polish=fb_p, held_polish=held_p, t_polish=round(time.time() - t1, 1))
    return out


def search_one(item):
    name, x0 = item
    t0 = time.time()
    cands, st = rs.search(np.array(x0, float), extra_layers=6, top_k=16)
    cands = sorted(cands, key=lambda c: (c[1], c[0]))[:32]
    return name, np.array(x0, float), cands, st, time.time() - t0


if __name__ == "__main__":
    q_ep1 = m.obs_to_state({"orientations": [0.843422702084278, 0.118726313648022, 0.5372505426789769, -0.992927017684357], "velocity": [0, 0]})
    starts = [
        ("ep1", q_ep1.tolist()),
        ("hanging", [np.pi - 0.01, 0.01, 0, 0]),
        ("side", [1.57, 0.2, 0, 0]),
        ("folded", [2.5, -2.5, 0, 0]),
        ("high", [0.6, -0.8, 0, 0]),
        ("low2", [-2.8, 1.0, 0, 0]),
    ]
    t0 = time.time()
    with mp.Pool(6) as pool:
        searched = pool.map(search_one, starts)
    jobs = []
    for name, x0, cands, st, dt in searched:
        print(name, "found_layer", st["found_layer"], "peak", st["peak"], "cands", len(cands), "search_s", round(dt, 1))
        for score, layer, seq in cands:
            jobs.append((name, x0, score, layer, seq))
    print("jobs", len(jobs), "search elapsed", round(time.time() - t0, 1), flush=True)
    with mp.Pool(48) as pool:
        results = pool.map(attempt, jobs, chunksize=1)
    print("total elapsed", round(time.time() - t0, 1))
    for name, *_ in searched:
        rows = [r for r in results if r["name"] == name]
        lm_ok = sum(r["held_lm"] for r in rows)
        pol_ok = sum(bool(r.get("held_polish")) for r in rows)
        best = max(rows, key=lambda r: max(r["ret_lm"], r["ret_polish"] or 0))
        print(f"{name}: attempts {len(rows)} lm_held {lm_ok} polish_held {pol_ok} "
              f"median_t_lm {np.median([r['t_lm'] for r in rows]):.1f}")
        print("   best", best)
        print("   lm rets", sorted([r["ret_lm"] for r in rows], reverse=True)[:8])
        print("   polish rets", sorted([r["ret_polish"] or 0 for r in rows], reverse=True)[:8], flush=True)
