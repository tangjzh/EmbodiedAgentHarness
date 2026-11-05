"""Spend a development reset and capture the initial state to dev/out/<CAPTURE_TAG>."""
import json
import os

import cv2
import numpy as np

import robot_api as robot

r = robot.reset()
print("reset returned:", json.dumps(r, default=str)[:3000])
TAG = "ep%s_step0" % robot.observe().get("episode")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", TAG)
os.makedirs(OUT, exist_ok=True)
obs = robot.observe()
cams = robot.cameras()
json.dump(obs, open(os.path.join(OUT, "obs.json"), "w"), indent=1)
for k, v in cams.items():
    np.save(os.path.join(OUT, k + ".npy"), v)
    if k.endswith("_rgb"):
        cv2.imwrite(os.path.join(OUT, k + ".png"), cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
print("episode", obs.get("episode"), "step", obs.get("native_step"), "remaining", obs.get("development_remaining"))
for s in ("left", "right"):
    print(s, np.round(obs["robot"][s]["eef_position_world"], 5).tolist())
