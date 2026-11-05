"""Offline mock of robot_api for dry-running controller logic: first-order EEF velocity model fitted on
episode-1 data, constant yaw rate, a crude bolt contact (nut cannot go below tip unless centred within
MOCK_CAPTURE (2 mm, ep06); then lands at tip-2.25 mm and descends 1.27 mm/rev). Cameras are the saved episode-1 step-0 frames."""
import copy
import json
import math

import numpy as np

_d = '/workspace/data/snap'
_obs0 = json.load(open(f'{_d}/obs.json'))
for n in _obs0['cameras']:
    _obs0['cameras'][n]['depth_pixel_ray_offset'] = [0.5, 0.5]
_cams = {k: np.load(f'{_d}/{k}.npy') for k in ('scene_rgb', 'scene_depth_m', 'closeup_rgb', 'closeup_depth_m', 'wrist_rgb', 'wrist_depth_m')}
import os
CAP = float(os.environ.get('MOCK_CAPTURE', '0.002')); MU = float(os.environ.get('MOCK_MU', '0.5'))
AX = np.array([0.553655, -0.002463]) + np.array(json.loads(os.environ.get('MOCK_AX_SHIFT', '[0,0]'))); TIP = 0.08485; OH = np.array([-1.455e-3, -0.048e-3])
S = dict(step=0, p=np.array(_obs0['robot']['eef_position_world'], float), v=np.zeros(3), psi=1.8005515, s=np.zeros(6),
         landed=False, succ=False)


def _quat(psi):
    # roll=pi, pitch=0, yaw=psi  -> R = Rz(psi) Rx(pi)
    return [0.0, math.cos(psi / 2), math.sin(psi / 2), 0.0]


def observe():
    o = copy.deepcopy(_obs0)
    o['robot']['eef_position_world'] = S['p'].tolist()
    o['robot']['eef_quaternion_wxyz'] = _quat(S['psi'])
    o['native_step'] = S['step']
    o['terminal'] = S['step'] >= 449
    o['native_success'] = S['succ']
    return o


def cameras():
    return _cams


def step(action, repeat=1):
    a = np.asarray(action, float)
    S['s'][:5] = 0.2 * a[:5] + 0.8 * S['s'][:5]
    gain = np.array([0.0115, 0.0115, 0.0068]) * (1 - np.array([0.9, 0.9, 0.87]))
    S['v'] = np.array([0.9, 0.9, 0.87]) * S['v'] + gain / (1 - np.array([0.9, 0.9, 0.87])) * (1 - np.array([0.9, 0.9, 0.87])) * S['s'][:3]
    contact = (not S['landed']) and S['p'][2] <= TIP + 0.0002 + 1e-6 and S['s'][2] < 0
    if contact:                                  # ep06: pressed on the tip the hand could not slide the nut
        sxy = np.linalg.norm(S['s'][:2]); lim = MU * abs(S['s'][2])
        S['v'][:2] *= 0.0 if sxy <= lim else (1 - lim / sxy)
    p = S['p'] + S['v']
    nut = p[:2] + np.array([math.cos(S['psi']) * OH[0] + math.sin(S['psi']) * OH[1], math.sin(S['psi']) * OH[0] - math.cos(S['psi']) * OH[1]])
    off = np.linalg.norm(nut - AX)
    if S['landed']:
        p[:2] = AX - (nut - p[:2])
        floor = TIP - 0.00225 - 0.00127 * (S['psi0'] - S['psi']) / (2 * math.pi)
        p[2] = max(p[2], floor) if p[2] > floor - 1 else floor
        p[2] = floor if p[2] < floor else p[2]
    elif p[2] < TIP + 0.0002:
        if off < CAP:
            S['landed'] = True; S['psi0'] = S['psi']
            p[2] = TIP - 0.00225
        elif off < 0.014:
            p[2] = TIP + 0.0002
            S['v'][2] = 0
    S['p'] = p
    S['psi'] -= math.radians(0.385)
    S['step'] += 1
    ny = S['psi'] % (2 * math.pi)
    ny = ny - 2 * math.pi if ny > math.radians(235) else ny
    if S['landed'] and ny < 0 and p[2] < TIP - 0.00225:
        S['succ'] = True
    o = observe()
    term = S['step'] >= 449
    return dict(observation=o, terminal=term, native_success=S['succ'], final_native_metric={'mock': True} if term else None)


def reset():
    raise RuntimeError('mock: reset not allowed')


def finish():
    raise RuntimeError('mock: finish not allowed')
