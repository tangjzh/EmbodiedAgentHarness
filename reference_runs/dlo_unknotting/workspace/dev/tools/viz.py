"""Offline composite view of a capture dir: rel top-down (2 mm) + scene crops around P + both wrist views.
usage: python3 tools/viz.py capdir [out.png]"""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, '/workspace')
import rope_vision as rv

d = sys.argv[1].rstrip('/') + '/'
out = sys.argv[2] if len(sys.argv) > 2 else d + 'viz.png'
obs = json.load(open(d + 'obs.json'))
cams = {}
for n in rv.CAMS:
    cams[n + '_rgb'] = np.load(d + n + '_rgb.npy')
    cams[n + '_depth_m'] = np.load(d + n + '_depth_m.npy')
P = np.array(obs['robot']['right']['eef_position_world'])
Q = np.array(obs['robot']['left']['eef_position_world'])
res = 0.002
x0, x1, y0, y1 = P[0] - 0.2, P[0] + 0.2, P[1] - 0.45, P[1] + 0.35
pts, _ = rv.rope_points(obs, cams)
zb = rv.height_map(pts, x0, x1, y0, y1, res)
top = rv.render_height(zb, 0.15, [(P[0], P[1]), (Q[0], Q[1])], x1, y0, res)   # 400x200 ; z colour 0..0.15 m
top = cv2.resize(top, (800, 400), interpolation=cv2.INTER_NEAREST)


def proj(name, X):
    K = np.array(obs['cameras'][name]['intrinsic_cv']); T = np.array(obs['cameras'][name]['cam2world_cv'])
    c = np.linalg.inv(T) @ np.append(X, 1)
    return K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]


tiles = []
for n in ('scene_left', 'scene_right'):
    rgb = cv2.cvtColor(cams[n + '_rgb'], cv2.COLOR_RGB2BGR)
    u, v = proj(n, P - np.array([0, 0, 0.0]))
    u0 = int(np.clip(u - 60, 0, 320 - 120)); v0 = int(np.clip(v - 40, 0, 240 - 100))
    tiles.append(cv2.resize(rgb[v0:v0 + 100, u0:u0 + 120], (480, 400), interpolation=cv2.INTER_CUBIC))
for n in ('wrist_right', 'wrist_left'):
    tiles.append(cv2.resize(cv2.cvtColor(cams[n + '_rgb'], cv2.COLOR_RGB2BGR), (533, 400)))
row1 = np.concatenate([top, tiles[0]], 1)
row2 = np.concatenate([tiles[1], tiles[2]], 1)
row3 = np.concatenate([tiles[3], np.zeros((400, 1280 - 533, 3), np.uint8)], 1)
row2 = cv2.resize(row2, (1280, int(400 * 1280 / row2.shape[1])))
img = np.concatenate([row1, row2], 0)
txt = 'step %s  P=(%.3f,%.3f,%.3f) Q=(%.3f,%.3f,%.3f)' % (obs.get('native_step'), *P, *Q)
cv2.putText(img, txt, (5, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.imwrite(out, img)
print('wrote', out)
