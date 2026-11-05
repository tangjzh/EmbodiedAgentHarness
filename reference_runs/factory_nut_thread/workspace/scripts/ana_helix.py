"""Offline: is the bolt thread a helix? Crest height (mod 2 mm) vs azimuth around the axis, per camera."""
import json
import math
import sys

import numpy as np

sys.path.insert(0, '/workspace')
import controller as C

obs = json.load(open('/workspace/data/snap/obs.json'))
cams = {k: np.load(f'/workspace/data/snap/{k}.npy') for k in ('scene_depth_m', 'closeup_depth_m', 'wrist_depth_m')}
AX = np.array([0.553655, -0.002463]); TIP = 0.08485
for name in ('wrist', 'closeup', 'scene'):
    pw = C._cam_points(cams, obs, name)
    d = pw[..., :2] - AX
    r = np.linalg.norm(d, axis=-1)
    z = pw[..., 2] - TIP
    m = np.isfinite(r) & (r < 0.0095) & (z > -0.0235) & (z < -0.003)
    az = np.degrees(np.arctan2(d[..., 1], d[..., 0]))[m]
    zz, rr = z[m] * 1e3, r[m] * 1e3
    print(name, 'n', m.sum(), 'azimuth range %.0f..%.0f' % (np.percentile(az, 2), np.percentile(az, 98)))
    for a0 in np.arange(-180, 180, 20):
        sel = (az >= a0) & (az < a0 + 20)
        if sel.sum() < 40:
            continue
        # crest points: radius > 7.85 mm; phase of their z modulo 2 mm (circular mean)
        cr = sel & (rr > 7.88)
        if cr.sum() < 5:
            continue
        ph = 2 * np.pi * zz[cr] / 2.0
        mean = math.atan2(np.sin(ph).mean(), np.cos(ph).mean())
        R = math.hypot(np.sin(ph).mean(), np.cos(ph).mean())
        print('  az %4d..%4d n %4d crest z mod 2mm = %.2f mm (R=%.2f)' % (a0, a0 + 20, cr.sum(), (mean / (2 * np.pi) * 2.0) % 2.0, R))
