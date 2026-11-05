"""Exp 1. System-identification run on the current episode (already started).

Hypothesis: the analytic model from public parameters predicts one-step and
short-horizon motion to near numerical precision, and the reward formula matches.
Excitation: piecewise-constant random actions so both joints move broadly.
Also inspects what robot.step returns and when the episode terminates.
"""
import json
import numpy as np
import robot_api as robot
import acrobot_model as m

rng = np.random.default_rng(1)
first = robot.observe()
print("observe:", first)
states = [m.obs_to_state(first)]
actions, rewards = [], []
u = 0.0
last = None
OBS_KEYS = ("observation", "observations", "obs")
REW_KEYS = ("reward", "rewards")


def unpack(res):
    """Pull the newest observation and reward out of a step result of unknown shape."""
    obs = next((res[k] for k in OBS_KEYS if k in res), res)
    if isinstance(obs, list):
        obs = obs[-1]
    rew = next((res[k] for k in REW_KEYS if k in res), np.nan)
    if isinstance(rew, (list, tuple)):
        rew = rew[-1]
    return obs, rew


for t in range(1500):
    if t % 25 == 0:
        u = float(rng.uniform(-1, 1))
    last = robot.step([u], 1)
    if t == 0:
        print("step type:", type(last).__name__)
        print("step sample:", last)
    o, r = unpack(last)
    actions.append(u)
    rewards.append(r)
    states.append(m.obs_to_state(o))
    if last.get("terminal"):
        print("terminal at tick", t + 1)
        break

X = np.array(states)
U = np.array(actions)
R = np.array(rewards, dtype=float)
np.savez("data_exp01.npz", X=X, U=U, R=R)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


pred = m.step(X[:-1], U)
err = X[1:] - pred
err[:, :2] = wrap(err[:, :2])
print("ticks", len(U))
print("one-step abs err max", np.abs(err).max(axis=0))
print("one-step abs err rms", np.sqrt((err ** 2).mean(axis=0)))

for H in (25, 100, 300):
    errs = []
    for s in range(0, len(U) - H, max(1, H // 2)):
        x = X[s].copy()
        for k in range(H):
            x = m.step(x, U[s + k])
        e = x - X[s + H]
        e[:2] = wrap(e[:2])
        errs.append(np.abs(e))
    errs = np.array(errs)
    print(f"H={H} open-loop err median {np.median(errs, axis=0)} max {errs.max(axis=0)}")

r_pred = m.reward(X[1:])
print("reward formula err max", np.nanmax(np.abs(r_pred - R)))
print("return", np.nansum(R), "| final step result:", last)
