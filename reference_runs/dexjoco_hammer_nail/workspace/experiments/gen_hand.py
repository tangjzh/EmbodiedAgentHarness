"""Rows that keep the flange target and ramp the hand to a target posture.

usage: gen_hand.py '<16 joint json>' n_ramp n_hold [flange_json_file_or_-]
Flange target: from output/grasp_plan.json (flange, rotvec) unless 'state' is given -> measured pose.
"""
import sys, json
sys.path.insert(0, 'experiments')
import numpy as np
from motion import *

hand1 = json.loads(sys.argv[1])
n_ramp, n_hold = int(sys.argv[2]), int(sys.argv[3])
src = sys.argv[4] if len(sys.argv) > 4 else 'plan'
state = json.load(open('output/state.json'))
p0, r0, h0 = state_pose(state)
if src == 'plan':
    gp = json.load(open('output/grasp_plan.json'))
    pos, rv = gp['flange'], gp['rotvec']
else:
    pos = list(p0)
    rv = list(continuous_rotvec(r0, np.array([-np.pi, 0, 0])))
h0 = np.array(h0)
h1 = np.array(hand1)
rows = []
for i in range(1, n_ramp + 1):
    s = i / n_ramp
    rows.append([float(v) for v in list(pos) + list(rv) + list((1 - s) * h0 + s * h1)])
rows += [rows[-1]] * n_hold
json.dump(rows, open('output/rows.json', 'w'))
print(len(rows), [round(v, 3) for v in rows[-1]])
