"""Offline Exp: reach search then iLQR landing on a few representative starts."""
import multiprocessing as mp
import time

import numpy as np

import acrobot_model as m
import reach_search as rs
import traj_opt as to

K_HOLD, P_HOLD = to.upright_lqr()


def land(x0, U_seed, hold_extra=120):
    """Stage 1: reach upright exactly. Stage 2: maximise native reward with a terminal LQR cost."""
    T = len(U_seed) + hold_extra
    U = np.concatenate([U_seed, np.zeros(hold_extra)])
    t_arr = len(U_seed)
    w_ang = np.zeros(T + 1)
    w_vel = np.zeros(T + 1)
    w_ang[t_arr:] = 20.0
    w_vel[t_arr:] = 2.0
    cost1 = to.Cost(T, 0.0, w_ang, w_vel, 1e-4, P_HOLD * 1e-2)
    s1 = to.ilqr(x0, U, cost1, iters=150)
    cost2 = to.Cost(T, 1.0, 0.0, 0.0, 1e-5, P_HOLD * 1e-2)
    s2 = to.ilqr(x0, s1["U"], cost2, iters=150)
    return s1, s2


def run(args):
    name, x0 = args
    t0 = time.time()
    x0 = np.array(x0, float)
    cands, st = rs.search(x0)
    t_search = time.time() - t0
    best = None
    tried = 0
    report = []
    # evaluate candidates in order of arrival layer, then near-miss score
    for score, layer, seq in sorted(cands, key=lambda c: (c[1], c[0]))[:8]:
        tried += 1
        s1, s2 = land(x0, seq)
        for label, s in (("s1", s1), ("s2", s2)):
            Ks = to.tvlqr(s["X"], s["U"], P_HOLD)
            res = to.closed_loop(x0, s["X"], s["U"], Ks, K_HOLD)
            report.append((label, layer, round(score, 3), round(res["ret"], 1)))
            if best is None or res["ret"] > best["ret"]:
                best = dict(ret=res["ret"], label=label, layer=layer, U=s["U"], X=s["X"], Ks=Ks, rewards=res["rewards"])
    first_ball = int(np.argmax(best["rewards"] >= 1.0)) + 1 if best and (best["rewards"] >= 1.0).any() else None
    return dict(name=name, x0=x0.tolist(), found_layer=st["found_layer"], peak=st["peak"], t_search=round(t_search, 1),
                t_total=round(time.time() - t0, 1), best_ret=None if best is None else round(best["ret"], 1),
                best_label=None if best is None else best["label"], best_layer=None if best is None else best["layer"],
                first_ball=first_ball, report=report[:16], n_cands=len(cands))


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
    with mp.Pool(len(starts)) as pool:
        for r in pool.imap_unordered(run, starts):
            print({k: v for k, v in r.items() if k != "report"})
            print("   ", r["report"], flush=True)
    print("elapsed", round(time.time() - t0, 1))
