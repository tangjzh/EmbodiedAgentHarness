"""Capture full-resolution RGB-D from all cameras + observation, save to dev/out/<tag>."""
import json
import os
import sys

import cv2
import numpy as np

import robot_api as robot

TAG = os.environ.get("CAPTURE_TAG", "ep1_step0")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", TAG)
os.makedirs(OUT, exist_ok=True)

obs = robot.observe()
cams = robot.cameras()
with open(os.path.join(OUT, "obs.json"), "w") as f:
    json.dump(obs, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
for k, v in cams.items():
    v = np.asarray(v)
    print(k, v.dtype, v.shape, float(np.nanmin(v)), float(np.nanmax(v)))
    np.save(os.path.join(OUT, k + ".npy"), v)
    if k.endswith("_rgb"):
        cv2.imwrite(os.path.join(OUT, k + ".png"), cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
print("keys", list(obs.keys()))
print("saved to", OUT)
