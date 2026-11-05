"""Real-plant recovery test: inside ONE dev episode, repeat {eject ball, random shake, hand over to policy} and time each
re-catch. Native return of these episodes is deliberately low and NOT comparable with normal episodes; the measurement
is per-trial re-catch latency from dynamic states the policy would otherwise see only after a failed catch."""
import importlib
import time
import numpy as np
import robot_api as robot


def run(exp, seed, module="controller", reset=True, timeout=200, hold_ok=25, par=None):
    ctrl = importlib.import_module(module)
    if reset:
        print("reset ->", robot.reset())
    obs = robot.observe()
    rng = np.random.default_rng(seed)
    mem = {}
    if par is not None:
        mem["par"] = par
    log, trials = [], []
    k, term, ret = 0, False, 0.0
    tpol = []

    def do(a):
        nonlocal obs, k, term, ret
        out = robot.step([float(a[0]), float(a[1])], repeat=1)
        obs = out["observation"]
        r = float(np.sum(out.get("rewards", 0.0)))
        term = bool(out.get("terminal", False))
        ret += r
        log.append([float(a[0]), float(a[1])] + list(obs["position"]) + list(obs["velocity"]) + [r])
        k += 1
        return r

    def in_cup():
        q = obs["position"]
        return abs(q[2] - q[0]) < 0.05 and -0.1 < (0.2 + q[3]) - (0.6 + q[1]) < 0.02

    first_trial = True
    while not term:
        if not first_trial:
            # eject: throw up, then whip down and sideways; repeat until the ball is out of the cup
            tries = 0
            while in_cup() and not term and tries < 4:
                sgn = rng.choice([-1.0, 1.0])
                for _ in range(14):
                    if not term:
                        do([0.0, 1.0])
                for _ in range(12):
                    if not term:
                        do([sgn, -1.0])
                tries += 1
        first_trial = False
        # random smooth shake with random duration, so the handover state differs between trials
        n_shake = int(rng.integers(5, 45))
        a = rng.uniform(-1, 1, 2)
        for i in range(n_shake):
            if term:
                break
            if i % 6 == 0:
                a = rng.uniform(-1, 1, 2)
            do(a)
        if term:
            break
        q0, v0 = list(obs["position"]), list(obs["velocity"])
        start_k = k
        first, streak, done = None, 0, False
        mem["stale"] = True
        while not term and k - start_k < timeout:
            t0 = time.time()
            act = ctrl.policy(obs, k, mem)
            tpol.append(time.time() - t0)
            r = do(act)
            if r > 0:
                if first is None:
                    first = k - start_k
                streak += 1
                if streak >= hold_ok:
                    done = True
                    break
            else:
                streak = 0
        if not term or done:
            trials.append({"t0": start_k, "lat": first, "ok": done, "handover": [round(x, 3) for x in q0 + v0]})
    L = np.array(log)
    np.savez(f"memory/exp{exp:02d}.npz", log=L, trials=np.array([str(t) for t in trials]), tpol=np.array(tpol))
    print(f"RESULT exp {exp} seed {seed}: ticks {k} native_return {ret:.0f} (not comparable) trials {len(trials)} "
          f"ok {sum(t['ok'] for t in trials)}")
    for t in trials:
        print(t)
    lats = [t["lat"] for t in trials if t["ok"]]
    if lats:
        print("re-catch latency ticks: median", int(np.median(lats)), "max", max(lats), "| policy ms median",
              round(float(np.median(tpol)) * 1000, 1), "max", round(float(np.max(tpol)) * 1000, 1))
    return trials
