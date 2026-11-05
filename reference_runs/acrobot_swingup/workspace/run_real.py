"""Run the current controller.py on one real development episode and report evidence.

Reads the experiment label from RUN_LABEL.txt so the same harness is reused.
Logs: start state, predicted vs real return, first tick in the ball, held at end,
per-episode one-step model error, policy compute time.
"""
import importlib
import json
import time

import numpy as np
import robot_api as robot

import acrobot_model as m
import controller

lines = open("RUN_LABEL.txt").read().split()
label = lines[0]
# "noreset" means the open episode has spent no ticks, so resetting would waste it.
if "noreset" in lines[1:]:
    print("using the already open episode, no reset")
else:
    print("reset returned:", robot.reset())
obs = robot.observe()
x0 = m.obs_to_state(obs)
print("start state", np.round(x0, 4).tolist())
memory = {}
X, U, R = [x0], [], []
compute = []
step = 0
last = None
while True:
    t0 = time.time()
    action = controller.policy(obs, step, memory)
    compute.append(time.time() - t0)
    last = robot.step(action, 1)
    obs = last["observation"]
    U.append(float(action[0]))
    R.append(float(last["rewards"][-1]))
    X.append(m.obs_to_state(obs))
    step += 1
    if last["terminal"]:
        break

X, U, R = np.array(X), np.array(U), np.array(R)
pred = m.step(X[:-1], U)
err = X[1:] - pred
err[:, :2] = (err[:, :2] + np.pi) % (2 * np.pi) - np.pi
in_ball = R >= 1.0
first = int(np.argmax(in_ball)) + 1 if in_ball.any() else None
plan = memory.get("plan", {})
summary = dict(
    label=label, ticks=len(U), x0=x0.tolist(), real_return=float(R.sum()), native_return=last["native_return"],
    predicted_return=float(plan.get("ret", np.nan)), plan_label=plan.get("label"), plan_info=plan.get("info"),
    first_ball=first, ball_ticks=int(in_ball.sum()), held_at_end=bool(in_ball[-1]),
    final_state=np.round(X[-1], 6).tolist(), model_err_max=float(np.abs(err).max()),
    first_call_s=round(compute[0], 2), max_later_call_ms=round(1000 * max(compute[1:]), 2), mean_later_call_ms=round(1000 * float(np.mean(compute[1:])), 3),
    remaining=last.get("development_remaining"),
)
print(json.dumps(summary, indent=1, default=float))
np.savez(f"data_{label}.npz", X=X, U=U, R=R)
with open("memory/real_runs.jsonl", "a") as f:
    f.write(json.dumps(summary, default=float) + "\n")
