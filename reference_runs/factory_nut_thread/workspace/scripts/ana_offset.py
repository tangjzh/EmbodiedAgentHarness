"""Offline: compare bolt-axis / nut-offset perception with pixel-ray offsets 0 and 0.5 against the
episode-1 thread-fit reference (axis (0.553655,-0.002463), nut offset in hand frame (-1.455,-0.048) mm)."""
import copy
import json
import math
import sys

import cv2
import numpy as np

sys.path.insert(0, '/workspace')
import controller as C

AX = np.array([0.553655, -0.002463])
OH = np.array([-1.455e-3, -0.048e-3])


def hand_axes(psi):
    return np.array([math.cos(psi), math.sin(psi)]), np.array([math.sin(psi), -math.cos(psi)])


def to_hand(off_w, psi):
    x, y = hand_axes(psi)
    return np.array([off_w @ x, off_w @ y])


def run(obs, cams, eef, psi, off):
    o = copy.deepcopy(obs)
    for n in o['cameras']:
        o['cameras'][n]['depth_pixel_ray_offset'] = [off, off]
    b = C.perceive_bolt(o, cams, eef)
    nut = C.perceive_nut_offset(o, cams, eef)
    line = f'off={off}: '
    if b:
        line += 'bolt err(mm) ' + ' '.join(f"{e['cam']}:({(e['c'][0]-AX[0])*1e3:+.2f},{(e['c'][1]-AX[1])*1e3:+.2f}) r={e['r']*1e3:.2f} tip={e['ztip']:.5f}" for e in b['ests'])
    line += ' | nut hand(mm) ' + ' '.join(f"{e['cam']}:({to_hand(np.asarray(e['off']), psi)[0]*1e3:+.2f},{to_hand(np.asarray(e['off']), psi)[1]*1e3:+.2f}) ztop={e['ztop']*1e3:.2f}" for e in nut['ests'])
    print(line)


# step-0 full dump
obs = json.load(open('/workspace/data/snap/obs.json'))
cams = {k: np.load(f'/workspace/data/snap/{k}.npy') for k in ('scene_rgb', 'scene_depth_m', 'closeup_rgb', 'closeup_depth_m', 'wrist_rgb', 'wrist_depth_m')}
eef = np.array(obs['robot']['eef_position_world'])
psi = C.yaw_of(obs['robot']['eef_quaternion_wxyz'])
print('snap step', obs['native_step'], 'psi', psi)
for off in (0.0, 0.5):
    run(obs, cams, eef, psi, off)

# episode-1 periodic snapshots (closeup + wrist only); camera poses: closeup static, wrist pose not logged
L = [json.loads(l) for l in open('/workspace/data/ep01/log.jsonl')]
steps = {l['step']: l for l in L if l['kind'] == 'step'}
for s in (30, 60):
    l = steps[s]
    c = {}
    for n in ('closeup', 'wrist'):
        c[f'{n}_rgb'] = cv2.cvtColor(cv2.imread(f'/workspace/data/ep01/s{s:03d}_{n}_rgb.png'), cv2.COLOR_BGR2RGB)
        c[f'{n}_depth_m'] = np.load(f'/workspace/data/ep01/s{s:03d}_{n}_depth.npy')
    print('ep01 step', s, '(closeup only meaningful: wrist pose unknown here)')
