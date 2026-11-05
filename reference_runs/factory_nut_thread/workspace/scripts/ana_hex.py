"""Offline: hex support-function fit of the nut top face (hand frame) — centre and orientation."""
import json
import math
import sys

import cv2
import numpy as np

sys.path.insert(0, '/workspace')
import controller as C

obs = json.load(open('/workspace/data/snap/obs.json'))
cams = {k: np.load(f'/workspace/data/snap/{k}.npy') for k in ('closeup_rgb', 'closeup_depth_m', 'wrist_rgb', 'wrist_depth_m')}
eef = np.array(obs['robot']['eef_position_world'])
psi = C.yaw_of(obs['robot']['eef_quaternion_wxyz'])
hx = np.array([math.cos(psi), math.sin(psi)]); hy = np.array([math.sin(psi), -math.cos(psi)])


def hexfit(T, phi0, q=99.7):
    A, b, w = [], [], []
    for k in range(3):
        a = phi0 + k * math.pi / 3
        n = np.array([math.cos(a), math.sin(a)])
        p = T @ n
        hi, lo = np.percentile(p, q), np.percentile(p, 100 - q)
        A.append(n); b.append((hi + lo) / 2); w.append(hi - lo)
    c = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    return c, np.array(w)


for name in ('wrist', 'closeup'):
    rgb = cams[f'{name}_rgb']
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yellow = (hsv[..., 1] > 100) & (hsv[..., 0] > 15) & (hsv[..., 0] < 40) & (hsv[..., 2] > 60)
    pw = C._cam_points(cams, obs, name)
    rel = pw - eef
    ok = yellow & np.isfinite(rel).all(-1) & (np.linalg.norm(rel[..., :2], axis=-1) < 0.03) & (np.abs(rel[..., 2]) < 0.03)
    P = rel[ok]
    zt = np.percentile(P[:, 2], 97)
    T = P[np.abs(P[:, 2] - zt) < 0.0005]
    T = np.stack([T[:, :2] @ hx, T[:, :2] @ hy], -1) * 1e3
    best = None
    for deg in np.arange(0, 60, 0.5):
        c, w = hexfit(T, math.radians(deg))
        if best is None or w.sum() < best[0]:
            best = (w.sum(), deg, c, w)
    print(name, 'ztop %.2f' % (zt * 1e3), 'best phi0 %.1f deg' % best[1], 'centre', np.round(best[2], 3), 'widths', np.round(best[3], 2))
    for deg in (30.0, 90.0 % 60):
        c, w = hexfit(T, math.radians(deg))
        print('   phi0=%g centre' % deg, np.round(c, 3), 'widths', np.round(w, 2))
    for q in (99.0, 99.5, 99.9, 100.0):
        c, w = hexfit(T, math.radians(30.0), q)
        print('   q=%g centre' % q, np.round(c, 3), 'widths', np.round(w, 2))
