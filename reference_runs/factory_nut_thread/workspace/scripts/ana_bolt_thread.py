"""Offline: bolt shank surface radius vs height (wrist + closeup, step-0 dump) to see thread ridges / pitch."""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, '/workspace')
import controller as C

obs = json.load(open('/workspace/data/snap/obs.json'))
cams = {k: np.load(f'/workspace/data/snap/{k}.npy') for k in ('scene_rgb', 'scene_depth_m', 'closeup_rgb', 'closeup_depth_m', 'wrist_rgb', 'wrist_depth_m')}
AX = np.array([0.553655, -0.002463]); TIP = 0.08485
for name in ('wrist', 'closeup'):
    pw = C._cam_points(cams, obs, name)
    r = np.linalg.norm(pw[..., :2] - AX, axis=-1)
    z = pw[..., 2] - TIP
    m = np.isfinite(r) & (r < 0.0095) & (z > -0.026) & (z < 0.0005)
    zz, rr = z[m] * 1e3, r[m] * 1e3
    print(name, 'n', m.sum())
    bins = np.arange(-25, 0.5, 0.25)
    idx = np.digitize(zz, bins)
    prof = []
    for i in range(1, len(bins)):
        sel = idx == i
        if sel.sum() > 3:
            prof.append((bins[i - 1], np.percentile(rr[sel], 90), sel.sum()))
    print(' '.join(f'{b:.2f}:{r:.2f}' for b, r, n in prof))
    # spectrum of r(z) to find the period
    P = np.array(prof)
    if len(P) > 20:
        zg = np.arange(P[0, 0], P[-1, 0], 0.25)
        rg = np.interp(zg, P[:, 0], P[:, 1])
        rg = rg - np.convolve(rg, np.ones(13) / 13, 'same')
        F = np.abs(np.fft.rfft(rg * np.hanning(len(rg))))
        fr = np.fft.rfftfreq(len(rg), 0.25)
        k = np.argmax(F[3:]) + 3
        print(' dominant period %.2f mm' % (1 / fr[k]))
