"""Offline helper: render top-down rope views from a saved capture directory."""
import json
import sys

import cv2
import numpy as np

d = sys.argv[1].rstrip('/') + '/'
x0, x1, y0, y1 = [float(a) for a in sys.argv[2:6]] if len(sys.argv) > 5 else (0.45, 0.95, -0.65, 0.15)
res = float(sys.argv[6]) if len(sys.argv) > 6 else 0.001
cams = sys.argv[7].split(',') if len(sys.argv) > 7 else ['scene_left', 'scene_right', 'wrist_left', 'wrist_right']
obs = json.load(open(d + 'obs.json'))
W = int((y1 - y0) / res); H = int((x1 - x0) / res)
zbuf = np.full((H, W), -1.0); img = np.zeros((H, W, 3), np.uint8)
for cam in cams:
    rgb = np.load(d + cam + '_rgb.npy'); dep = np.load(d + cam + '_depth_m.npy')
    K = np.array(obs['cameras'][cam]['intrinsic_cv']); T = np.array(obs['cameras'][cam]['cam2world_cv'])
    hh, ww = dep.shape
    u, v = np.meshgrid(np.arange(ww) + 0.5, np.arange(hh) + 0.5)
    P = np.stack([(u - K[0, 2]) / K[0, 0] * dep, (v - K[1, 2]) / K[1, 1] * dep, dep, np.ones_like(dep)], -1) @ T.T
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(int)
    m = (hsv[..., 0] >= 85) & (hsv[..., 0] < 100) & (hsv[..., 1] >= 20) & (hsv[..., 1] < 120) & (hsv[..., 2] >= 40)
    p = P[m]
    r = ((x1 - p[:, 0]) / res).astype(int); c = ((p[:, 1] - y0) / res).astype(int)
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    for rr, cc, zz in zip(r[ok], c[ok], p[ok, 2]):
        if zz > zbuf[rr, cc]:
            zbuf[rr, cc] = zz
valid = zbuf > -0.5
zn = np.clip(zbuf / 0.06, 0, 1)
col = cv2.applyColorMap((zn * 255).astype(np.uint8), cv2.COLORMAP_JET)
img[valid] = col[valid]
for side, c in [('left', (255, 255, 255)), ('right', (255, 255, 255))]:
    e = obs['robot'][side]['eef_position_world']
    cv2.drawMarker(img, (int((e[1] - y0) / res), int((x1 - e[0]) / res)), c, cv2.MARKER_CROSS, 12, 1)
out = d + 'topdown_%s.png' % ('_'.join(c[:2] for c in cams))
cv2.imwrite(out, img)
print(out, img.shape, 'z max', zbuf.max())
