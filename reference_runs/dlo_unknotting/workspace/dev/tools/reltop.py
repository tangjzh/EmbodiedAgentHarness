"""Offline: top-down height map of rope in coordinates relative to the RIGHT EEF (P) for capture dirs.
usage: python3 tools/reltop.py outfile.png capdir1 [capdir2 ...]   (images stacked vertically)
"""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, '/workspace')
import rope_vision as rv

RES = 0.002
DX0, DX1, DY0, DY1 = -0.22, 0.22, -0.48, 0.32
rows = []
for d in sys.argv[2:]:
    d = d.rstrip('/') + '/'
    obs = json.load(open(d + 'obs.json'))
    cams = {}
    for n in rv.CAMS:
        cams[n + '_rgb'] = np.load(d + n + '_rgb.npy')
        cams[n + '_depth_m'] = np.load(d + n + '_depth_m.npy')
    P = np.array(obs['robot']['right']['eef_position_world'])
    Q = np.array(obs['robot']['left']['eef_position_world'])
    pts, _ = rv.rope_points(obs, cams)
    x0, x1, y0, y1 = P[0] + DX0, P[0] + DX1, P[1] + DY0, P[1] + DY1
    zb = rv.height_map(pts, x0, x1, y0, y1, RES)
    img = rv.render_height(zb, 0.06, [(P[0], P[1]), (Q[0], Q[1])], x1, y0, RES)
    cv2.putText(img, d.split('out/')[-1], (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    rows.append(img)
cv2.imwrite(sys.argv[1], np.concatenate(rows, 0))
print('wrote', sys.argv[1])
