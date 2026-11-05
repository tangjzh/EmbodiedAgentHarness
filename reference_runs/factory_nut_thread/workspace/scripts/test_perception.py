"""Offline: run controller perception on saved step-0 dumps (ep01 snap and ep02 s000) and compare to thread fits."""
import json
import math
import sys

import cv2
import numpy as np

sys.path.insert(0, '/workspace')
import controller as C


def run(obs, cams, ref_axis=None, ref_tip=None):
    eef = C._eef(obs)
    psi = C.yaw_of(obs['robot']['eef_quaternion_wxyz'])
    nut = C.perceive_nut(obs, cams, eef, psi)
    zb = nut['ztop'] - C.NUT_H
    bolt = C.perceive_bolt(obs, cams, eef, zb)
    print(' nut o_hand mm', np.round(nut['o_hand'] * 1e3, 3), 'src', nut['src'], 'ztop %.3f zb %.3f mm' % (nut['ztop'] * 1e3, zb * 1e3))
    print(' bolt xy', np.round(bolt['xy'], 6), 'tip %.5f' % bolt['ztip'])
    for e in bolt['ests']:
        print('   ', e['cam'], np.round(np.array(e['c']) * 1e3, 3), 'r %.2f mm rms %.3f tip %.5f n %d' % (e['r'] * 1e3, e['rms'] * 1e3, e['ztip'], e['n']),
              'd_ref mm', np.round((np.array(e['c']) - ref_axis) * 1e3, 3) if ref_axis is not None else '')


obs = json.load(open('/workspace/data/snap/obs.json'))
cams = {}
for k in ('scene', 'closeup', 'wrist'):
    cams[f'{k}_rgb'] = np.load(f'/workspace/data/snap/{k}_rgb.npy')
    cams[f'{k}_depth_m'] = np.load(f'/workspace/data/snap/{k}_depth_m.npy')
print('ep01')
run(obs, cams, np.array([0.553655, -0.002463]))
L = [json.loads(l) for l in open('/workspace/data/ep02/log.jsonl')]
obs = [l for l in L if l['kind'] == 'obs0'][0]['obs']
cams = {}
for k in ('scene', 'closeup', 'wrist'):
    cams[f'{k}_rgb'] = cv2.cvtColor(cv2.imread(f'/workspace/data/ep02/s000_{k}_rgb.png'), cv2.COLOR_BGR2RGB)
    cams[f'{k}_depth_m'] = np.load(f'/workspace/data/ep02/s000_{k}_depth.npy')
print('ep02')
run(obs, cams, np.array([0.555208, 0.025119]))
