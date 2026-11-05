"""Nut top-face world height (wrist + closeup depth) vs EEF z over an episode's snapshots: detects grasp slip
and measures the nut's own descent relative to the bolt tip."""
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
O = [l for l in L if l['kind'] == 'obs0']
steps = {l['step']: l for l in L if l['kind'] == 'step'}
tip = P['bolt']['ztip']
cams0 = (O[0]['obs'] if O else json.load(open('/workspace/data/snap/obs.json')))['cameras']
# wrist pose moves with the hand: cam2world at step s = T_hand(s) @ inv(T_hand(0)) @ T_wrist(0)
eef0 = np.array(P['eef']); q0 = (O[0]['obs'] if O else json.load(open('/workspace/data/snap/obs.json')))['robot']['eef_quaternion_wxyz']


def hand_T(eef, q):
    T = np.eye(4); T[:3, :3] = C.quat_to_R(q); T[:3, 3] = eef; return T


Tw0 = np.array(cams0['wrist']['cam2world_cv']); Th0 = hand_T(eef0, q0)
for f in sorted(glob.glob(f'{d}/s*_wrist_depth.npy')):
    s = int(re.findall(r's(\d+)_', os.path.basename(f))[0])
    if s not in steps and s != 0:
        continue
    l = steps.get(s)
    eef = np.array(l['eef']) if l else eef0
    out = []
    for name in ('wrist', 'closeup'):
        rgb = cv2.cvtColor(cv2.imread(f'{d}/s{s:03d}_{name}_rgb.png'), cv2.COLOR_BGR2RGB)
        dep = np.load(f'{d}/s{s:03d}_{name}_depth.npy')
        K = np.array(cams0[name]['intrinsic_cv'])
        if name == 'wrist':
            if l is None:
                T = Tw0
            else:
                # rebuild the hand orientation from the logged yaw (roll=pi, pitch=0)
                psi = l['yaw']; c, sn = math.cos(psi), math.sin(psi)
                R = np.array([[c, sn, 0], [sn, -c, 0], [0, 0, -1.0]])
                Th = np.eye(4); Th[:3, :3] = R; Th[:3, 3] = eef
                T = Th @ np.linalg.inv(Th0) @ Tw0
        else:
            T = np.array(cams0[name]['cam2world_cv'])
        pw = C.backproject(dep, K, T)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        yellow = (hsv[..., 1] > 100) & (hsv[..., 0] > 15) & (hsv[..., 0] < 40) & (hsv[..., 2] > 60)
        rel = pw - eef
        ok = yellow & np.isfinite(rel).all(-1) & (np.linalg.norm(rel[..., :2], axis=-1) < 0.03) & (np.abs(rel[..., 2]) < 0.03)
        if ok.sum() < 50:
            out.append('%s -' % name); continue
        z = pw[ok][:, 2]
        zt = np.percentile(z, 97)
        out.append('%s top-tip %.3f top-eef %.3f' % (name, (zt - tip) * 1e3, (zt - eef[2]) * 1e3))
    print('s%03d eef-tip %.3f | %s' % (s, (eef[2] - tip) * 1e3, ' | '.join(out)))
