"""Nut pose in the HAND frame from each wrist snapshot (camera rigid on the hand): top-face plane tilt, face
height, hex centre/orientation. Detects tilt/slip of the nut in the grasp. usage: ana_nuttilt.py <tag>"""
import glob
import json
import math
import os
import re
import sys

import cv2
import numpy as np

sys.modules.setdefault('robot_api', type(sys)('robot_api'))
sys.path.insert(0, '/workspace')
import controller as C

tag = sys.argv[1]
d = f'/workspace/data/{tag}'
L = [json.loads(l) for l in open(f'{d}/log.jsonl')]
P = [l for l in L if l['kind'] == 'perception'][0]
O = [l for l in L if l['kind'] == 'obs0'][0]['obs']
steps = {l['step']: l for l in L if l['kind'] == 'step'}
eef0 = np.array(P['eef']); psi0 = P['psi']


def hand_T(eef, psi):
    c, s = math.cos(psi), math.sin(psi)
    T = np.eye(4); T[:3, :3] = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.0]]); T[:3, 3] = eef; return T


X = np.linalg.inv(hand_T(eef0, psi0)) @ np.array(O['cameras']['wrist']['cam2world_cv'])
K = np.array(O['cameras']['wrist']['intrinsic_cv'])
for f in sorted(glob.glob(f'{d}/s*_wrist_depth.npy')):
    s = int(re.findall(r's(\d+)_', os.path.basename(f))[0])
    l = steps.get(s)
    rgb = cv2.cvtColor(cv2.imread(f'{d}/s{s:03d}_wrist_rgb.png'), cv2.COLOR_BGR2RGB)
    ph = C.backproject(np.load(f), K, X)
    ok = C._yellow(rgb) & np.isfinite(ph).all(-1) & (np.linalg.norm(ph[..., :2], axis=-1) < 0.03)
    Pp = ph[ok]
    zt = np.percentile(-Pp[:, 2], 97)
    F = Pp[np.abs(-Pp[:, 2] - zt) < 0.0008]
    # plane fit z = a x + b y + c (hand frame), robust: 2 passes
    for _ in range(2):
        A = np.c_[F[:, 0], F[:, 1], np.ones(len(F))]
        sol = np.linalg.lstsq(A, F[:, 2], rcond=None)[0]
        r = F[:, 2] - A @ sol
        F = F[np.abs(r) < 3 * np.std(r) + 1e-5]
    tilt = math.degrees(math.atan(math.hypot(sol[0], sol[1])))
    az = math.degrees(math.atan2(sol[1], sol[0]))
    ctr, phi0, w = C._hex_fit(F[:, :2])
    print('s%03d %-8s h %+6.2f | tilt %.2f deg az %6.1f | face z(hand,ctr) %.3f mm | ctr %s phi0 %.2f w %s' % (
        s, l['phase'] if l else 'init', (l['h'] * 1e3) if l else float('nan'), tilt, az,
        -(sol[0] * ctr[0] + sol[1] * ctr[1] + sol[2]) * 1e3, np.round(ctr * 1e3, 3), phi0, np.round(w * 1e3, 2)))
