"""Preview a vertical step of size dz from the current measured state for several row counts.

usage: preview_step.py dz n1 n2 ...
"""
import json
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')
import numpy as np
from api.runtime import Context
from motion import meta_of, state_pose, continuous_rotvec

dz = float(sys.argv[1])
ns = [int(v) for v in sys.argv[2:]]
state = json.load(open('output/state.json'))
last = json.load(open('output/last_target.json'))
pos, rv, hand = np.array(last['pos']), last['rotvec'], last['hand']
with Context() as ctx:
    for n in ns:
        # settle first at last target for 30 rows then step
        rows = [[float(v) for v in list(pos) + list(rv) + list(hand)]] * 30
        low = pos + np.array([0, 0, dz])
        rows += [[float(v) for v in list(low) + list(rv) + list(hand)]] * n
        rep = ctx.call('dexjoco_preview_run', actions=rows)
        m = meta_of(rep)
        es = m['end_state']
        print(n, 'end z', round(es[2], 4), 'dz moved', round(es[2] - pos[2], 4))
