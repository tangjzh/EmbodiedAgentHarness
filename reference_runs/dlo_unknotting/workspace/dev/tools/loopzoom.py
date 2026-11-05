"""Zoomed scene-camera crops around P's hanging loop for several capture dirs.
usage: python3 tools/loopzoom.py out.png capdir...  (crop centred at P with z=zc)"""
import json
import sys

import cv2
import numpy as np

out = sys.argv[1]
dirs = sys.argv[2:]
zc = 0.17
if dirs and dirs[0].startswith('zc='):  # zc=<float> absolute, zc=rel -> max(0.1, P_z-0.2)
    zc = dirs.pop(0)[3:]


def proj(obs, name, X):
    K = np.array(obs['cameras'][name]['intrinsic_cv']); T = np.array(obs['cameras'][name]['cam2world_cv'])
    c = np.linalg.inv(T) @ np.append(X, 1)
    return K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]


rows = []
for cam in ['scene_left', 'scene_right']:
    tiles = []
    for s in dirs:
        d = s.rstrip('/') + '/'
        obs = json.load(open(d + 'obs.json'))
        P = np.array(obs['robot']['right']['eef_position_world'])
        M = P.copy(); M[2] = max(0.1, P[2] - 0.2) if zc == 'rel' else float(zc)
        u, v = proj(obs, cam, M)
        a = cv2.imread(d + cam + '_rgb.png')
        u0 = int(np.clip(u - 40, 0, 320 - 80)); v0 = int(np.clip(v - 50, 0, 240 - 100))
        t = cv2.resize(a[v0:v0 + 100, u0:u0 + 80], (320, 400), interpolation=cv2.INTER_CUBIC)
        cv2.putText(t, '/'.join(d.split('/')[-3:-1]) + ' ' + cam[6:], (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(t)
    rows.append(np.concatenate(tiles, 1))
cv2.imwrite(out, np.concatenate(rows, 0))
print('wrote', out)
