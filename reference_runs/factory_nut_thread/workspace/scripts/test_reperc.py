"""Offline re-perception check on saved snapshots: perceive_bolt with the known tip (current controller) vs the
thread-fit axis.  usage: python3 scripts/test_reperc.py <tag> [snap ...]   (snap = logged step of the image)"""
import json
import math
import sys

import cv2
import numpy as np

sys.modules['robot_api'] = type(sys)('robot_api')
sys.path.insert(0, '/workspace')
import controller as C

tag = sys.argv[1]
D = f'/workspace/data/{tag}'
L = [json.loads(l) for l in open(f'{D}/log.jsonl')]
P = [l for l in L if l['kind'] == 'perception'][0]
O = [l for l in L if l['kind'] == 'obs0'][0]['obs']
S = {l['step']: l for l in L if l['kind'] == 'step'}
T = [l for l in S.values() if l['phase'] == 'thread']
A, b = [], []
for l in T[len(T) // 3:]:
    c, s = math.cos(l['yaw']), math.sin(l['yaw'])
    A += [[1, 0, -c, -s], [0, 1, -s, c]]
    b += [l['eef'][0], l['eef'][1]]
x = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0] if A else np.full(4, np.nan)
ax_true = x[:2]
tip, zb = P['bolt']['ztip'], P['zb']
print(tag, 'thread-fit axis (mm)', np.round(ax_true * 1e3, 3), 'tip', round(tip, 5), 'zb mm', round(zb * 1e3, 2))


def T_of(p, R):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, p
    return M


e0 = np.array(O['robot']['eef_position_world'])
R0 = C.quat_to_R(O['robot']['eef_quaternion_wxyz'])
Tw0 = np.array(O['cameras']['wrist']['cam2world_cv'])
snaps = [int(a) for a in sys.argv[2:]] or sorted(k for k in S if k % 30 == 0 and k > 0)
for n in snaps:
    if n not in S:
        continue
    l = S[n]
    c, s = math.cos(l['yaw']), math.sin(l['yaw'])
    R = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.0]])
    eef = np.array(l['eef'])
    obs = json.loads(json.dumps(O))
    obs['cameras']['wrist']['cam2world_cv'] = (T_of(eef, R) @ np.linalg.inv(T_of(e0, R0)) @ Tw0).tolist()
    cams = {}
    try:
        for nm in ('closeup', 'wrist'):
            cams[f'{nm}_rgb'] = cv2.cvtColor(cv2.imread(f'{D}/s{n:03d}_{nm}_rgb.png'), cv2.COLOR_BGR2RGB)
            cams[f'{nm}_depth_m'] = np.load(f'{D}/s{n:03d}_{nm}_depth.npy')
    except Exception:
        continue
    cams['scene_rgb'] = np.zeros((240, 320, 3), np.uint8)
    cams['scene_depth_m'] = np.zeros((240, 320), np.float32)
    hb = l['h'] * 1e3
    off = (np.array(l['eef'][:2]) + C.hand_to_world_xy(x[2:], l['yaw']) - ax_true) * 1e3
    for mode, kw in (('old', {}), ('tip', dict(tip=tip))):
        if mode == 'old':
            bb = None
            import importlib
            V = sys.modules.get('ctrl_v33')
            if V is None:
                import importlib.util
                spec = importlib.util.spec_from_file_location('ctrl_v33', '/workspace/scripts/controller_v33.py')
                V = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(V)
                sys.modules['ctrl_v33'] = V
            bb = V.perceive_bolt(obs, cams, eef, zb)
        else:
            bb = C.perceive_bolt(obs, cams, eef, zb, **kw)
        ests = [] if bb is None else bb['ests']
        txt = ' '.join('%s(%+.2f,%+.2f r%.2f n%d)' % (e['cam'][:2], *((np.array(e['c']) - ax_true) * 1e3), e['r'] * 1e3, e['n'])
                       for e in ests)
        print('  s%03d %-7s %s hb %+.2f nut-axis %.2f mm | %s %s' % (n, l['phase'][:7], mode, hb, np.linalg.norm(off), txt, '' if ests else 'NONE'))
