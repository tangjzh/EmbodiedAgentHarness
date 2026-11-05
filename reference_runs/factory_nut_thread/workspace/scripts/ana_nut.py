"""Offline: study nut top-face geometry from the wrist camera at episode-1 step 0 (hand frame, offset 0.5)."""
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
for name in ('wrist', 'closeup'):
    rgb = cams[f'{name}_rgb']
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yellow = (hsv[..., 1] > 100) & (hsv[..., 0] > 15) & (hsv[..., 0] < 40) & (hsv[..., 2] > 60)
    pw = C._cam_points(cams, obs, name)
    rel = pw - eef
    ok = yellow & np.isfinite(rel).all(-1) & (np.linalg.norm(rel[..., :2], axis=-1) < 0.03) & (np.abs(rel[..., 2]) < 0.03)
    z = rel[ok][:, 2]
    h, e = np.histogram(z * 1e3, bins=np.arange(-2, 16, 0.25))
    print(name, 'n yellow', ok.sum())
    print(' z hist (mm):', ' '.join(f'{e[i]:.2f}:{h[i]}' for i in range(len(h)) if h[i] > 20))
    P = rel[ok]
    hxy = np.stack([P[:, :2] @ hx, P[:, :2] @ hy], -1) * 1e3
    top = np.abs(P[:, 2] * 1e3 - np.percentile(P[:, 2] * 1e3, 97)) < 0.5
    T = hxy[top]
    print(' top n', len(T), 'hand x range', T[:, 0].min(), T[:, 0].max(), 'y range', T[:, 1].min(), T[:, 1].max())
    # occupancy image of top-face points in hand frame, 0.1 mm grid, +-16 mm
    g = np.zeros((321, 321), np.uint8)
    ij = np.round((T + 16) / 0.1).astype(int)
    m = (ij >= 0).all(1) & (ij < 321).all(1)
    g[ij[m, 1], ij[m, 0]] = 255
    g = cv2.dilate(g, np.ones((3, 3), np.uint8))
    cv2.imwrite(f'/tmp/top_{name}.png', cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    # all yellow points occupancy (side faces too), any z
    g2 = np.zeros((321, 321), np.uint8)
    ij = np.round((hxy + 16) / 0.1).astype(int)
    m = (ij >= 0).all(1) & (ij < 321).all(1)
    g2[ij[m, 1], ij[m, 0]] = 255
    cv2.imwrite(f'/tmp/all_{name}.png', cv2.resize(g2, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
