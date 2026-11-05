"""Offline: where is the nut bottom relative to the EEF? Use closeup depth at ep01 snapshots where the nut sits
on the bolt: highest visible shank point (non-yellow, near axis) vs EEF z."""
import json
import math
import sys

import cv2
import numpy as np

sys.path.insert(0, '/workspace')
import controller as C

obs = json.load(open('/workspace/data/snap/obs.json'))
K = np.array(obs['cameras']['closeup']['intrinsic_cv']); T = np.array(obs['cameras']['closeup']['cam2world_cv'])
AX = np.array([0.553655, -0.002463]); TIP = 0.08485
L = [json.loads(l) for l in open('/workspace/data/ep01/log.jsonl')]
steps = {l['step']: l for l in L if l['kind'] == 'step'}
for s in (60, 90, 120, 150, 180, 210, 240):
    rgb = cv2.cvtColor(cv2.imread(f'/workspace/data/ep01/s{s:03d}_closeup_rgb.png'), cv2.COLOR_BGR2RGB)
    d = np.load(f'/workspace/data/ep01/s{s:03d}_closeup_depth.npy')
    pw = C.backproject(d, K, T)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yellow = (hsv[..., 1] > 100) & (hsv[..., 0] > 15) & (hsv[..., 0] < 40) & (hsv[..., 2] > 60)
    r = np.linalg.norm(pw[..., :2] - AX, axis=-1)
    eef = np.array(steps[s]['eef'])
    shank = (~yellow) & np.isfinite(r) & (r < 0.0085) & (pw[..., 2] > TIP - 0.024) & (pw[..., 2] < TIP + 0.01)
    ys = yellow & np.isfinite(r) & (r < 0.016)
    zs = pw[shank][:, 2]
    zy = pw[ys][:, 2]
    print(s, 'eef-tip %.2f mm' % ((eef[2] - TIP) * 1e3), 'shank top (rel tip) %.2f mm' % ((np.percentile(zs, 99) - TIP) * 1e3) if len(zs) else '-',
          'n', len(zs), '| yellow z range rel eef: %.2f..%.2f mm' % ((np.percentile(zy, 1) - eef[2]) * 1e3, (np.percentile(zy, 99) - eef[2]) * 1e3) if len(zy) else '')
