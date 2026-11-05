"""Run one REAL development episode with a controller module, exactly as an evaluator would call policy(). Logs to memory/."""
import importlib
import time
import numpy as np
import robot_api as robot


def run(exp, module="controller", reset=True, max_ticks=1000, par=None):
    ctrl = importlib.import_module(module)
    if reset:
        print("reset ->", robot.reset())
    obs = robot.observe()
    print("exp", exp, "module", module, "start obs", obs)
    mem = {}
    if par is not None:
        mem["par"] = par
    log, modes, tpol = [], [], []
    ret, first, k, term = 0.0, None, 0, False
    while k < max_ticks and not term:
        t0 = time.time()
        a = ctrl.policy(obs, k, mem)
        tpol.append(time.time() - t0)
        out = robot.step([float(a[0]), float(a[1])], repeat=1)
        obs = out["observation"]
        r = float(np.sum(out.get("rewards", 0.0)))
        term = bool(out.get("terminal", False))
        ret += r
        if r > 0 and first is None:
            first = k
        log.append([float(a[0]), float(a[1])] + list(obs["position"]) + list(obs["velocity"]) + [r])
        modes.append(mem.get("mode", "?"))
        k += 1
    L = np.array(log)
    np.savez(f"memory/exp{exp:02d}.npz", log=L, modes=np.array(modes), tpol=np.array(tpol))
    Q = L[:, 2:6]
    lx = Q[:, 2] - Q[:, 0]
    lz = (0.2 + Q[:, 3]) - (0.6 + Q[:, 1])
    tp = np.array(tpol) * 1000
    rew = L[:, -1] > 0
    changes = [(i, modes[i]) for i in range(len(modes)) if i == 0 or modes[i] != modes[i - 1]]
    print(f"RESULT exp {exp}: ticks {k} return {ret:.0f} first_reward_tick {first} rewarded {int(rew.sum())} "
          f"maxlz {lz.max():.3f} policy ms median {np.median(tp):.1f} max {tp.max():.1f} total {tp.sum()/1000:.1f}s")
    print("mode changes (first 30):", changes[:30], "n_changes", len(changes))
    lost = [i for i in range(1, len(rew)) if rew[i - 1] and not rew[i]]
    print("reward-lost ticks (first 15):", lost[:15], "n", len(lost))
    n = first if first is not None else min(k, 120)
    print("tick ux uz | cx cz bx bz | lx lz mode")
    for i in list(range(0, min(n + 12, k), 4)):
        print(f"{i:3d} {L[i,0]:+.2f} {L[i,1]:+.2f} | {Q[i,0]:+.3f} {Q[i,1]:+.3f} {Q[i,2]:+.3f} {Q[i,3]:+.3f} | "
              f"{lx[i]:+.3f} {lz[i]:+.3f} {modes[i]} {'R' if rew[i] else ''}")
    return ret, first
