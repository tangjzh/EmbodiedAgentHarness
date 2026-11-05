"""Experiment 1: system ID. Zero action, x/z steps, then resonant x sine. Logs to memory/exp01.npz."""
import numpy as np
import robot_api as robot

first = robot.observe()
print("observe keys:", sorted(first.keys()))
print("start obs:", first)


def parse(out):
    """Return (obs, reward, terminal) from whichever shape robot.step uses."""
    if isinstance(out, dict):
        obs = out.get("observation", out.get("observations", out))
        r = out.get("reward", out.get("rewards", 0.0))
        term = bool(out.get("terminal", False))
    else:
        obs, r, term = out[0], out[1], bool(out[2])
    r = float(np.sum(r)) if r is not None else 0.0
    return obs, r, term


def schedule(k):
    t = k / 50.0
    if t < 0.5:
        return [0.0, 0.0]
    if t < 1.5:
        return [1.0, 0.0]
    if t < 2.5:
        return [-1.0, 0.0]
    if t < 3.5:
        return [0.0, 1.0]
    if t < 4.5:
        return [0.0, -1.0]
    if t < 10.5:
        return [float(np.sin(2 * np.pi * 0.91 * (t - 4.5))), 0.0]
    return [0.0, 0.0]


log = []
printed = False
k = 0
ret = 0.0
while k < 1100:
    a = schedule(k)
    out = robot.step(a, repeat=1)
    if not printed:
        print("step return type:", type(out).__name__)
        print("step raw:", out if not isinstance(out, dict) else {kk: out[kk] for kk in out})
        printed = True
    obs, r, term = parse(out)
    ret += r
    log.append(list(a) + list(obs["position"]) + list(obs["velocity"]) + [r])
    k += 1
    if term:
        print("terminal at policy step", k, "return", ret)
        break

L = np.array(log)
np.savez("memory/exp01.npz", log=L)
print("steps", len(L), "return", ret)

t = np.arange(len(L)) / 50.0
pos = L[:, 2:6]
print("t  a_x a_z  cup_x cup_z ball_x ball_z  string_len")
for i in list(range(0, min(len(L), 550), 25)) + list(range(550, len(L), 50)):
    cx, cz, bx, bz = pos[i]
    # world: cup anchor (cx, 0.6+cz-0.108); ball (bx, 0.2+bz)
    s = np.hypot(bx - cx, (0.2 + bz) - (0.492 + cz))
    print(f"{t[i]:5.2f} {L[i,0]:+.2f} {L[i,1]:+.2f}  {cx:+.3f} {cz:+.3f} {bx:+.3f} {bz:+.3f}  {s:.3f}")
s_all = np.hypot(pos[:, 2] - pos[:, 0], (0.2 + pos[:, 3]) - (0.492 + pos[:, 1]))
print("max string", s_all.max(), "ball world z max", (0.2 + pos[:, 3]).max(), "rim z at max", (0.6 + pos[:, 1]).max())
print("reward steps", int((L[:, -1] > 0).sum()))
