"""Controller v1: plan once at the first tick, then track with TVLQR and hold with LQR."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import acrobot_model as m  # noqa: E402
import reach_search as rs  # noqa: E402
import switch_land as sl  # noqa: E402
import traj_opt as to  # noqa: E402

K_HOLD, P_HOLD = to.upright_lqr()


def _hold_segment(x_end, extra):
    x = x_end.copy()
    us = np.zeros(extra)
    for t in range(extra):
        xw = x.copy()
        xw[:2] = to.wrap(xw[:2])
        us[t] = float(np.clip(-K_HOLD @ xw, -1, 1))
        x = m.step(x, us[t])
    return us


def _score(x0, X, U):
    Ks = to.tvlqr(X, U, P_HOLD)
    res = to.closed_loop(x0, X, U, Ks, K_HOLD)
    return res["ret"], Ks


def make_plan(x0, budget=45.0):
    t0 = time.time()
    info = {}
    # a start already inside the balancer's reach needs no swing-up
    empty = np.zeros((0, 4))
    hold_ret = to.closed_loop(x0, empty, np.zeros(0), empty, K_HOLD)["ret"]
    best = dict(ret=hold_ret, X=np.zeros((0, 4)), U=np.zeros(0), Ks=np.zeros((0, 4)), label="hold")
    cands, st = rs.search(x0, extra_layers=6, top_k=16)
    info["found_layer"] = st["found_layer"]
    info["t_search"] = time.time() - t0
    cands = sorted(cands, key=lambda c: (c[1], c[0]))
    n_ok = 0
    for score, layer, seq in cands:
        if time.time() - t0 > budget * 0.6 or n_ok >= 6:
            break
        lm = sl.land_switches(x0, seq, P_HOLD, iters=120, target=2.0)
        ret, Ks = _score(x0, lm["X"], lm["U"])
        if ret > best["ret"]:
            best = dict(ret=ret, X=lm["X"], U=lm["U"], Ks=Ks, label="lm", layer=layer)
        n_ok += int(lm["V"] < 2000.0)
    info["t_lm"] = time.time() - t0
    if best["label"] == "lm":
        U0 = np.concatenate([best["U"], _hold_segment(best["X"][-1], 150)])
        cost = to.Cost(len(U0), 1.0, 0.0, 0.0, 1e-5, P_HOLD * 1e-2)
        s = to.ilqr(x0, U0, cost, iters=80, deadline=t0 + budget)
        ret, Ks = _score(x0, s["X"], s["U"])
        if ret > best["ret"]:
            best = dict(ret=ret, X=s["X"], U=s["U"], Ks=Ks, label="polish", layer=best.get("layer"))
    info["t_total"] = time.time() - t0
    best["info"] = info
    return best


def policy(observation, step, memory):
    x = m.obs_to_state(observation)
    if "plan" not in memory:
        memory["plan"] = make_plan(x)
        memory["t"] = 0
    plan = memory["plan"]
    t = memory["t"]
    memory["t"] = t + 1
    if t < len(plan["U"]):
        dx = x - plan["X"][t]
        dx[:2] = to.wrap(dx[:2])
        u = plan["U"][t] - plan["Ks"][t] @ dx
    else:
        xw = x.copy()
        xw[:2] = to.wrap(xw[:2])
        u = -K_HOLD @ xw
    return [float(np.clip(u, -1.0, 1.0))]
