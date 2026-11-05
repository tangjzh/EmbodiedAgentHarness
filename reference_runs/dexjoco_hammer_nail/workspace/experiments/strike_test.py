"""Strike test: send a step target below the nail, one row per call, logging measured flange z.

usage: strike_test.py dz_total n_rows retract_dz retract_rows
Reads output/last_target.json (pos, rotvec, hand). Writes output/strike_log.json.
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
rdz = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
rn = int(sys.argv[4]) if len(sys.argv) > 4 else 0
last = json.load(open('output/last_target.json'))
pos, rv, hand = np.array(last['pos']), last['rotvec'], last['hand']
low = pos + np.array([0, 0, dz])
log = []
with Context() as ctx:
    for i in range(n):
        row = [float(v) for v in list(low) + list(rv) + list(hand)]
        t0 = time.time()
        rep = ctx.call('dexjoco_step', actions=[row])
        m = meta_of(rep)
        st = m['state']
        log.append({'i': i, 'z': st[2], 'xy': st[0:2], 'hand': st[7:23], 'ended': m.get('episode_ended'),
                    'complete': m.get('task_complete'), 'feedback': (m.get('transition') or {}).get('feedback'), 'dt': time.time() - t0})
        print(i, round(st[2], 4), 'vz', round((st[2] - (log[-2]['z'] if len(log) > 1 else pos[2])) / 0.02, 3) if True else '', m.get('task_complete'), (m.get('transition') or {}).get('feedback'))
        if m.get('episode_ended'):
            break
    if rn > 0:
        up = low + np.array([0, 0, rdz])
        rows = []
        for i in range(1, rn + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * i / rn)
            p = (1 - s) * low + s * up
            rows.append([float(v) for v in list(p) + list(rv) + list(hand)])
        rep = ctx.call('dexjoco_step', actions=rows)
        m = meta_of(rep)
        print('after retract z', round(m['state'][2], 4), m.get('task_complete'))
        json.dump(m['state'], open('output/state.json', 'w'))
        last['pos'] = up.tolist()
        json.dump(last, open('output/last_target_pending.json', 'w'))
json.dump(log, open('output/strike_log.json', 'w'))
