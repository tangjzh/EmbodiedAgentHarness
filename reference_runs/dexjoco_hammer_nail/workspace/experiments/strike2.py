"""Strike with contact detection: step target dz below, one row per call; on stall, retract.

usage: strike2.py dz max_rows retract_dz retract_rows [stall_v]
"""
import json
import sys
import time

sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')
import numpy as np
from api.runtime import Context
from motion import meta_of

dz = float(sys.argv[1])
n = int(sys.argv[2])
rdz = float(sys.argv[3])
rn = int(sys.argv[4])
stall_v = float(sys.argv[5]) if len(sys.argv) > 5 else 0.08
last = json.load(open('output/last_target.json'))
pos, rv, hand = np.array(last['pos']), last['rotvec'], last['hand']
low = pos + np.array([0, 0, dz])
log = []
state0 = json.load(open('output/state.json'))
zprev = state0[2]
vmax = 0
with Context() as ctx:
    for i in range(n):
        row = [float(v) for v in list(low) + list(rv) + list(hand)]
        rep = ctx.call('dexjoco_step', actions=[row])
        m = meta_of(rep)
        st = m['state']
        v = (st[2] - zprev) / 0.02
        zprev = st[2]
        vmax = min(vmax, v)
        tr = m.get('transition') or {}
        log.append({'i': i, 'z': st[2], 'v': v, 'hand': st[7:23], 'complete': m.get('task_complete'), 'tr': {k: tr[k] for k in tr if k != 'steps'}})
        print(i, 'z', round(st[2], 4), 'vz', round(v, 3), 'complete', m.get('task_complete'), 'ended', m.get('episode_ended'))
        if m.get('episode_ended'):
            break
        if vmax < -0.2 and v > -stall_v:
            print('stall detected')
            break
    up = pos.copy()
    up[2] = low[2] + rdz if rdz < 9 else pos[2]
    rows = []
    for i in range(1, rn + 1):
        s = 0.5 - 0.5 * np.cos(np.pi * i / rn)
        p = (1 - s) * np.array([pos[0], pos[1], st[2]]) + s * up
        rows.append([float(v) for v in list(p) + list(rv) + list(hand)])
    rep = ctx.call('dexjoco_step', actions=rows)
    m = meta_of(rep)
    print('after retract z', round(m['state'][2], 4), 'complete', m.get('task_complete'), 'ended', m.get('episode_ended'))
    print('hand after', [round(v, 3) for v in m['state'][7:23]])
    json.dump(m['state'], open('output/state.json', 'w'))
    last['pos'] = up.tolist()
    json.dump(last, open('output/last_target.json', 'w'))
json.dump(log, open('output/strike_log.json', 'w'))
print(json.dumps({k: v for k, v in m.items() if k in ('episode_limits', 'task_complete', 'episode_ended')}))
