"""Offline: side view (y horizontal, z vertical) of rope points, coloured by camera. usage: sideview.py capdir [out]"""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, '/workspace')
import rope_vision as rv

d = sys.argv[1].rstrip('/') + '/'
out = sys.argv[2] if len(sys.argv) > 2 else d + 'side.png'
obs = json.load(open(d + 'obs.json'))
cams = {}
for n in rv.CAMS:
    cams[n + '_rgb'] = np.load(d + n + '_rgb.npy')
    cams[n + '_depth_m'] = np.load(d + n + '_depth_m.npy')
P = np.array(obs['robot']['right']['eef_position_world'])
Q = np.array(obs['robot']['left']['eef_position_world'])
pts, ci = rv.rope_points(obs, cams)
res = 0.002
y0, y1, z1 = P[1] - 0.45, P[1] + 0.3, 0.45
H, W = int(z1 / res), int((y1 - y0) / res)
img = np.zeros((H, W, 3), np.uint8)
colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255), (0, 255, 255)]  # BGR: sl blue, sr green, wl red, wr yellow
for k in range(4):
    p = pts[ci == k]
    r = ((z1 - p[:, 2]) / res).astype(int); c = ((p[:, 1] - y0) / res).astype(int)
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    img[r[ok], c[ok]] = colors[k]
for E in (P, Q):
    cv2.drawMarker(img, (int((E[1] - y0) / res), int((z1 - E[2]) / res)), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
for zz in (0.05, 0.1, 0.2, 0.3, 0.4):
    cv2.line(img, (0, int((z1 - zz) / res)), (8, int((z1 - zz) / res)), (255, 255, 255), 1)
cv2.putText(img, 'step %s  y:[P-0.45,P+0.3] z:[0,0.45]' % obs.get('native_step'), (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
cv2.imwrite(out, cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
print('wrote', out)
