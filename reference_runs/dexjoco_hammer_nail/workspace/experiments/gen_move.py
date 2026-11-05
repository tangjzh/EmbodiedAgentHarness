"""Generate rows from a given start state to a palm-down pose with yaw; optional pitch.

usage: gen_move.py tx ty tz yaw_deg n_rows [pitch_deg] [hold_rows] [state_json]
"""
import sys, json
sys.path.insert(0, 'experiments')
import numpy as np
from scipy.spatial.transform import Rotation as R
from motion import *

tx, ty, tz, yaw, n = [float(a) for a in sys.argv[1:6]]
pitch = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
hold = int(sys.argv[7]) if len(sys.argv) > 7 else 10
state_file = sys.argv[8] if len(sys.argv) > 8 else 'output/state.json'
state = json.load(open(state_file))
p0, r0, h0 = state_pose(state)
r1 = R.from_euler('z', np.radians(yaw)) * R.from_euler('y', np.radians(pitch)) * R.from_rotvec([np.pi, 0, 0])
near = continuous_rotvec(r0, np.array([-np.pi, 0, 0]))
hand = h0 if len(sys.argv) <= 9 else json.loads(sys.argv[9])
rows = interp_rows(p0, r0, h0, np.array([tx, ty, tz]), r1, hand, int(n), near=near)
rows += [rows[-1]] * hold
json.dump(rows, open('output/rows.json', 'w'))
print(len(rows), [round(v, 4) for v in rows[-1][:6]])
