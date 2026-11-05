"""Rows that move the flange target by a delta (world) keeping orientation and a given hand target.

usage: gen_delta.py dx dy dz n hold '<hand json>|keep' [from=plan|state|last]
"""
import sys, json
sys.path.insert(0, 'experiments')
import numpy as np
from motion import *

dx, dy, dz = [float(v) for v in sys.argv[1:4]]
n, hold = int(sys.argv[4]), int(sys.argv[5])
hand_arg = sys.argv[6]
src = sys.argv[7] if len(sys.argv) > 7 else 'last'
state = json.load(open('output/state.json'))
p0, r0, h0 = state_pose(state)
if src == 'last':
    last = json.load(open('output/last_target.json'))
    pos, rv, hand_prev = np.array(last['pos']), np.array(last['rotvec']), last['hand']
elif src == 'plan':
    gp = json.load(open('output/grasp_plan.json'))
    pos, rv, hand_prev = np.array(gp['flange']), np.array(gp['rotvec']), h0
else:
    pos, rv, hand_prev = p0, continuous_rotvec(r0, np.array([-np.pi, 0, 0])), h0
hand = hand_prev if hand_arg == 'keep' else json.loads(hand_arg)
target = pos + np.array([dx, dy, dz])
rows = []
for i in range(1, n + 1):
    s = 0.5 - 0.5 * np.cos(np.pi * i / n)
    p = (1 - s) * pos + s * target
    rows.append([float(v) for v in list(p) + list(rv) + list(hand)])
rows += [rows[-1]] * hold
json.dump(rows, open('output/rows.json', 'w'))
json.dump({'pos': target.tolist(), 'rotvec': list(map(float, rv)), 'hand': list(map(float, hand))}, open('output/last_target_pending.json', 'w'))
print(len(rows), [round(v, 3) for v in rows[-1]])
