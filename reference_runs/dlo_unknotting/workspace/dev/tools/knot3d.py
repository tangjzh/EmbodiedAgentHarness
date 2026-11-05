"""Offline: zoomed y-z and x-z views of rope points in a box around (P_x, P_y, zc). usage: knot3d.py out.png zc half capdir..."""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, '/workspace')
import rope_vision as rv

out, zc, hw = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rows = []
for d in sys.argv[4:]:
    d = d.rstrip('/') + '/'
    obs = json.load(open(d + 'obs.json'))
    cams = {}
    for n in rv.CAMS:
        cams[n + '_rgb'] = np.load(d + n + '_rgb.npy')
        cams[n + '_depth_m'] = np.load(d + n + '_depth_m.npy')
    P = np.array(obs['robot']['right']['eef_position_world'])
    pts, ci = rv.rope_points(obs, cams)
    res = hw / 120
    tiles = []
    for ax in (1, 0):
        c0 = P[ax] - hw
        img = np.zeros((240, 240, 3), np.uint8)
        colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255), (0, 255, 255)]
        oth = 1 - ax
        sel = np.abs(pts[:, oth] - P[oth]) < hw
        for k in range(4):
            p = pts[(ci == k) & sel]
            r = ((zc + hw - p[:, 2]) / res).astype(int); c = ((p[:, ax] - c0) / res).astype(int)
            ok = (r >= 0) & (r < 240) & (c >= 0) & (c < 240)
            img[r[ok], c[ok]] = colors[k]
        cv2.putText(img, '%s s%s' % ('yz' if ax == 1 else 'xz', obs.get('native_step')), (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    rows.append(np.concatenate(tiles, 1))
cv2.imwrite(out, np.concatenate(rows, 0))
print('wrote', out)
