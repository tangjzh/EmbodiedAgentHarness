"""Waypoint rows: list of (pos, rotvec-from-yaw/roll) segments from the last target.

usage: gen_wps.py '<json list of [x,y,z,yaw_deg,roll_deg,n]>' hold '<hand json>|keep'
roll is applied about the finger axis after yaw (palm-down base).
"""
import sys, json
sys.path.insert(0, 'experiments')
import numpy as np
from scipy.spatial.transform import Rotation as R
from motion import *

wps = json.loads(sys.argv[1])
hold = int(sys.argv[2])
hand_arg = sys.argv[3]
last = json.load(open('output/last_target.json'))
p0, rv0 = np.array(last['pos']), np.array(last['rotvec'])
hand = last['hand'] if hand_arg == 'keep' else json.loads(hand_arg)
r0 = R.from_rotvec(rv0)
rows = []
near = rv0
for (x, y, z, yaw, roll, n) in wps:
    r1 = R.from_euler('z', np.radians(yaw)) * R.from_rotvec([np.pi, 0, 0]) * R.from_euler('x', np.radians(roll))
    seg = interp_rows(p0, r0, hand, np.array([x, y, z]), r1, hand, int(n), near=near)
    rows += seg
    p0, r0, near = np.array([x, y, z]), r1, np.array(seg[-1][3:6])
rows += [rows[-1]] * hold
json.dump(rows, open('output/rows.json', 'w'))
json.dump({'pos': p0.tolist(), 'rotvec': rows[-1][3:6], 'hand': list(map(float, hand))}, open('output/last_target_pending.json', 'w'))
print(len(rows), [round(v, 4) for v in rows[-1][:6]])
