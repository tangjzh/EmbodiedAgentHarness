"""Offline Exp E: run the whole of controller.make_plan + closed loop over many starts."""
import multiprocessing as mp
import time

import numpy as np

import acrobot_model as m
import controller as c
import traj_opt as to


def run(x0):
    x0 = np.array(x0, float)
    t0 = time.time()
    try:
        plan = c.make_plan(x0)
        res = to.closed_loop(x0, plan["X"], plan["U"], plan["Ks"], c.K_HOLD)
    except Exception as exc:  # a crash here must be visible, not silently averaged away
        return dict(x0=x0.tolist(), ret=-1.0, held=False, first=None, secs=time.time() - t0, label="EXC " + repr(exc))
    r = res["rewards"]
    return dict(x0=x0.tolist(), ret=float(res["ret"]), held=bool(r[-1] >= 1.0), first=int(np.argmax(r >= 1.0)) + 1 if (r >= 1).any() else None,
                secs=time.time() - t0, label=plan["label"])


if __name__ == "__main__":
    rng = np.random.default_rng(2024)
    starts = [np.array([a, b, 0.0, 0.0]) for a, b in rng.uniform(-np.pi, np.pi, size=(200, 2))]
    edge = [
        [0.0, 0.0], [1e-4, -1e-4], [0.02, 0.02], [0.1, -0.1], [0.2, 0.3], [-0.4, 0.5],
        [np.pi, 0.0], [-np.pi, 0.0], [np.pi - 1e-3, 1e-3], [np.pi, np.pi], [0.0, np.pi], [np.pi, -np.pi],
        [3.14, 3.14], [-3.14, -3.14], [3.14, -3.14], [1.57, 1.57], [-1.57, -1.57], [1.57, -1.57], [-1.57, 1.57],
        [0.0, 1.57], [0.0, -3.0], [3.0, 3.0], [2.9, -0.1], [-2.9, 0.1],
    ]
    n_rand = len(starts)
    starts += [np.array([a, b, 0.0, 0.0]) for a, b in edge]
    t0 = time.time()
    with mp.Pool(64) as pool:
        out = pool.map(run, starts, chunksize=1)
    print("elapsed", round(time.time() - t0, 1))
    rand, ed = out[:n_rand], out[n_rand:]
    rets = np.array([o["ret"] for o in rand])
    print(f"random {n_rand}: mean {rets.mean():.1f} median {np.median(rets):.1f} p10 {np.percentile(rets, 10):.1f} min {rets.min():.1f} max {rets.max():.1f}")
    print("held", np.mean([o["held"] for o in rand]), "plan secs mean", np.mean([o["secs"] for o in rand]), "max", np.max([o["secs"] for o in rand]))
    print("labels", {l: sum(o["label"] == l for o in rand) for l in set(o["label"] for o in rand)})
    print("worst random:")
    for o in sorted(rand, key=lambda o: o["ret"])[:8]:
        print("  ", np.round(o["x0"][:2], 3).tolist(), round(o["ret"], 1), o["first"], o["held"], o["label"], round(o["secs"], 1))
    print("edge cases:")
    for o in ed:
        print("  ", np.round(o["x0"][:2], 4).tolist(), round(o["ret"], 1), o["first"], o["held"], o["label"], round(o["secs"], 1))
