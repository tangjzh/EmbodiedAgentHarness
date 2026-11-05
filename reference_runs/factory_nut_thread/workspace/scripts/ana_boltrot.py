"""Does the bolt rotate/move? Hex fit of the bolt head's top face (closeup + scene, world frame) per snapshot."""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, '/workspace')
import controller as C

tag = sys.argv[1]
d = f'/workspace/data/{tag}'
L = [json.loads(l) for l in open(f'{d}/log.jsonl')]
P = [l for l in L if l['kind'] == 'perception'][0]
O = [l for l in L if l['kind'] == 'obs0'][0]['obs']
ax = np.array(P['bolt']['xy']); tip = P['bolt']['ztip']
cam = O['cameras']['closeup']
K, T = np.array(cam['intrinsic_cv']), np.array(cam['cam2world_cv'])
# find the head top level once from s000: histogram of z in the annulus 9..14 mm
for f in sorted(glob.glob(f'{d}/s*_closeup_depth.npy')):
    s = int(re.findall(r's(\d+)_', os.path.basename(f))[0])
    pw = C.backproject(np.load(f), K, T)
    r = np.linalg.norm(pw[..., :2] - ax, axis=-1)
    m = np.isfinite(r) & (r > 0.0095) & (r < 0.016) & (pw[..., 2] < tip - 0.015) & (pw[..., 2] > tip - 0.04)
    z = pw[m][:, 2]
    if s == 0:
        hist, edges = np.histogram(z, bins=np.arange(tip - 0.04, tip - 0.015, 0.0002))
        zh = edges[np.argmax(hist)] + 0.0001
        print('head top level rel tip %.2f mm' % ((zh - tip) * 1e3))
    sel = m & (np.abs(pw[..., 2] - zh) < 0.0004)
    Pt = pw[sel][:, :2] - ax
    ctr, phi0, w = C._hex_fit(Pt, q=99.5)
    print('s%03d n %4d head hex phi0 %6.2f ctr mm %s widths %s' % (s, len(Pt), phi0, np.round(ctr * 1e3, 3), np.round(w * 1e3, 2)))
