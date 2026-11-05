"""Tight zoom at a world point (default: below P at z=zc) in both scene cams + wrist_right.
usage: python3 tools/knotzoom.py capdir out.png [zc] [half_px]"""
import json
import sys

import cv2
import numpy as np

d = sys.argv[1].rstrip('/') + '/'
out = sys.argv[2]
zc = float(sys.argv[3]) if len(sys.argv) > 3 else 0.11
hp = int(sys.argv[4]) if len(sys.argv) > 4 else 20
obs = json.load(open(d + 'obs.json'))
P = np.array(obs['robot']['right']['eef_position_world'])


def proj(name, X):
    K = np.array(obs['cameras'][name]['intrinsic_cv']); T = np.array(obs['cameras'][name]['cam2world_cv'])
    c = np.linalg.inv(T) @ np.append(X, 1)
    return K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]


tiles = []
for cam in ['scene_left', 'scene_right']:
    M = P.copy(); M[2] = zc
    u, v = proj(cam, M)
    a = cv2.imread(d + cam + '_rgb.png')
    a = cv2.copyMakeBorder(a, hp, hp, hp, hp, cv2.BORDER_CONSTANT)
    u0 = int(u); v0 = int(v)
    tiles.append(cv2.resize(a[v0:v0 + 2 * hp, u0:u0 + 2 * hp], (480, 480), interpolation=cv2.INTER_CUBIC))
w = cv2.imread(d + 'wrist_right_rgb.png')
tiles.append(cv2.resize(w, (640, 480), interpolation=cv2.INTER_CUBIC))
cv2.imwrite(out, np.concatenate(tiles, 1))
print('wrote', out)
