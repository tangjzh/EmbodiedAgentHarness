"""Zoomed scene-camera crops centred at P + (0, dy, 0) with absolute z for several capture dirs.
usage: python3 tools/cropzoom.py out.png dy=<m> z=<m> half=<px> capdir..."""
import json
import sys

import cv2
import numpy as np

out = sys.argv[1]
args = sys.argv[2:]
opt = {"dy": -0.15, "z": 0.03, "half": 45}
while args and "=" in args[0]:
    k, v = args.pop(0).split("=")
    opt[k] = float(v)


def proj(obs, name, X):
    K = np.array(obs['cameras'][name]['intrinsic_cv']); T = np.array(obs['cameras'][name]['cam2world_cv'])
    c = np.linalg.inv(T) @ np.append(X, 1)
    return K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]


h = int(opt["half"])
rows = []
for cam in ['scene_left', 'scene_right']:
    tiles = []
    for s in args:
        d = s.rstrip('/') + '/'
        obs = json.load(open(d + 'obs.json'))
        P = np.array(obs['robot']['right']['eef_position_world'])
        M = np.array([P[0], P[1] + opt["dy"], opt["z"]])
        u, v = proj(obs, cam, M)
        a = cv2.imread(d + cam + '_rgb.png')
        u0 = int(np.clip(u - h, 0, 320 - 2 * h)); v0 = int(np.clip(v - h, 0, 240 - 2 * h))
        t = cv2.resize(a[v0:v0 + 2 * h, u0:u0 + 2 * h], (360, 360), interpolation=cv2.INTER_CUBIC)
        cv2.putText(t, '/'.join(d.split('/')[-3:-1]) + ' ' + cam[6:], (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(t)
    rows.append(np.concatenate(tiles, 1))
cv2.imwrite(out, np.concatenate(rows, 0))
print('wrote', out)
