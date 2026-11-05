"""Hex orientation (phi0, hand frame) and centre of the nut top face in each wrist snapshot: rotational slip?"""
import glob
import json
import math
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, '/workspace')
import controller as C

tag = sys.argv[1]
d = f'/workspace/data/{tag}'
L = [json.loads(l) for l in open(f'{d}/log.jsonl')]
P = [l for l in L if l['kind'] == 'perception'][0]
O = [l for l in L if l['kind'] == 'obs0'][0]['obs']
steps = {l['step']: l for l in L if l['kind'] == 'step'}
cams0 = O['cameras']
eef0 = np.array(P['eef']); psi0 = P['psi']


def hand_T(eef, psi):
    c, s = math.cos(psi), math.sin(psi)
    T = np.eye(4); T[:3, :3] = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.0]]); T[:3, 3] = eef; return T


Tw0 = np.array(cams0['wrist']['cam2world_cv'])
X = np.linalg.inv(hand_T(eef0, psi0)) @ Tw0          # wrist camera in hand frame (constant)
K = np.array(cams0['wrist']['intrinsic_cv'])
for f in sorted(glob.glob(f'{d}/s*_wrist_depth.npy')):
    s = int(re.findall(r's(\d+)_', os.path.basename(f))[0])
    l = steps.get(s)
    eef, psi = (np.array(l['eef']), l['yaw']) if l else (eef0, psi0)
    rgb = cv2.cvtColor(cv2.imread(f'{d}/s{s:03d}_wrist_rgb.png'), cv2.COLOR_BGR2RGB)
    dep = np.load(f)
    # points directly in the hand frame
    ph = C.backproject(dep, K, X)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yellow = (hsv[..., 1] > 100) & (hsv[..., 0] > 15) & (hsv[..., 0] < 40) & (hsv[..., 2] > 60)
    ok = yellow & np.isfinite(ph).all(-1) & (np.linalg.norm(ph[..., :2], axis=-1) < 0.03)
    Pp = ph[ok]
    # hand z points down (R[2,2]=-1): top face is the most negative hand z
    zt = np.percentile(-Pp[:, 2], 97)
    T = Pp[np.abs(-Pp[:, 2] - zt) < 0.0005][:, :2]
    ctr, phi0, w = C._hex_fit(T)
    print('s%03d yaw %7.2f  phi0 %6.2f  ctr mm %s widths %s n %d' % (s, math.degrees(psi), phi0, np.round(ctr * 1e3, 3), np.round(w * 1e3, 2), len(T)))
