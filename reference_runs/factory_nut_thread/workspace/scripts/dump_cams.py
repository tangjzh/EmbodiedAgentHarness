"""Dump current full-resolution RGB-D frames and observation to /workspace/data/<tag>/ (no physics advance)."""
import json
import os
import sys

import cv2
import numpy as np
import robot_api as robot

tag = os.environ.get('DUMP_TAG', 'snap')
out = f'/workspace/data/{tag}'
os.makedirs(out, exist_ok=True)
obs = robot.observe()
cams = robot.cameras()
with open(f'{out}/obs.json', 'w') as f:
    json.dump(obs, f, indent=1)
for k, v in cams.items():
    np.save(f'{out}/{k}.npy', v)
    if k.endswith('_rgb'):
        cv2.imwrite(f'{out}/{k}.png', cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
    print(k, v.shape, v.dtype, float(np.nanmin(v)), float(np.nanmax(v)))
print('step', obs['native_step'], 'eef', obs['robot']['eef_position_world'], obs['robot']['eef_quaternion_wxyz'])
