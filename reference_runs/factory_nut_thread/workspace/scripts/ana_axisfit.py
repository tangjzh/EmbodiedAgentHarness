"""Fit bolt axis + nut hand-frame offset from threaded-phase EEF xy: eef = axis - R(psi) o_hand."""
import json
import math
import sys

import numpy as np

sys.path.insert(0, '/workspace')
import controller as C

tag = sys.argv[1]
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
L = [json.loads(l) for l in open(f'/workspace/data/{tag}/log.jsonl')]
P = [l for l in L if l['kind'] == 'perception'][0]
S = [l for l in L if l['kind'] == 'step' and l['phase'] == 'thread' and l['step'] >= lo]
A, b = [], []
for l in S:
    psi = l['yaw']
    c, s = math.cos(psi), math.sin(psi)
    # eef = axis - [c*o0 + s*o1, s*o0 - c*o1]
    A.append([1, 0, -c, -s]); b.append(l['eef'][0])
    A.append([0, 1, -s, c]); b.append(l['eef'][1])
A, b = np.array(A), np.array(b)
x, *_ = np.linalg.lstsq(A, b, rcond=None)
r = A @ x - b
print(tag, 'n', len(S), 'axis', np.round(x[:2], 6), 'o_hand mm', np.round(x[2:] * 1e3, 3), 'rms %.3f mm' % (np.sqrt(np.mean(r ** 2)) * 1e3))
print('  perceived axis', np.round(P['bolt']['xy'], 6), 'diff mm', np.round((np.array(P['bolt']['xy']) - x[:2]) * 1e3, 3))
o = P['nut'].get('o_hand', P['nut'].get('off'))
print('  perceived o_hand mm', np.round(np.array(o) * 1e3, 3), 'diff', np.round((np.array(o) - x[2:]) * 1e3, 3))
for e in P['bolt']['ests']:
    print('   bolt', e['cam'], 'diff mm', np.round((np.array(e['c']) - x[:2]) * 1e3, 3))
for e in P['nut']['ests']:
    oo = e.get('o_hand', e.get('off'))
    print('   nut', e['cam'], 'diff mm', np.round((np.array(oo) - x[2:]) * 1e3, 3))
# nut-centre position over the thread phase, relative to the fitted axis
nc = np.array([np.array(l['eef'][:2]) + C.hand_to_world_xy(x[2:], l['yaw']) - x[:2] for l in S]) * 1e3
print('  nut centre - axis (mm): mean', np.round(nc.mean(0), 3), 'max |.|', np.round(np.abs(nc).max(), 3))
