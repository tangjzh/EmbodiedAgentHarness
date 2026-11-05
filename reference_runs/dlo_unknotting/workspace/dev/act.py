"""Execute a list of actions from dev/cmd.json, capturing cameras after selected steps.

cmd.json: {"tag": str, "actions": [[12 floats, repeat], ...], "capture_every": int (0=only end)}
"""
import json
import os

import cv2
import numpy as np

import robot_api as robot

HERE = os.path.dirname(os.path.abspath(__file__))
cmd = json.load(open(os.path.join(HERE, "cmd.json")))
tag = cmd["tag"]
cap_every = int(cmd.get("capture_every", 0))
log = []


def capture(name):
    out = os.path.join(HERE, "out", tag, name)
    os.makedirs(out, exist_ok=True)
    obs = robot.observe()
    cams = robot.cameras()
    json.dump(obs, open(os.path.join(out, "obs.json"), "w"), indent=1)
    for k, v in cams.items():
        np.save(os.path.join(out, k + ".npy"), v)
        if k.endswith("_rgb"):
            cv2.imwrite(os.path.join(out, k + ".png"), cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
    return obs


n = 0
res = None
for i, (a, rep) in enumerate(cmd["actions"]):
    res = robot.step(a, repeat=rep)
    n += 1
    o = res.get("observation", res)
    ent = dict(i=i, step=o.get("native_step"), L=o["robot"]["left"]["eef_position_world"],
               R=o["robot"]["right"]["eef_position_world"], terminal=res.get("terminal", o.get("terminal")),
               metric=res.get("final_native_metric", o.get("final_native_metric")))
    log.append(ent)
    print(json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, list) else v) for k, v in ent.items()}))
    if cap_every and (n % cap_every == 0):
        capture("s%03d" % o.get("native_step"))
    if ent["terminal"]:
        print("TERMINAL", ent["metric"])
        break
capture("final")
json.dump(log, open(os.path.join(HERE, "out", tag, "log.json"), "w"))
print("keys of step result:", list(res.keys()) if res else None)
