"""Build pre-grasp and grasp rows from handle estimate.

usage: plan_grasp.py cx cy cz_top dirx diry slope a_g [pre_up] [n_move] [n_desc]
"""
import sys, json
sys.path.insert(0, 'experiments')
import numpy as np
from motion import *
from grasp import *

cx, cy, cz, dx, dy, slope, a_g = [float(v) for v in sys.argv[1:8]]
pre_up = float(sys.argv[8]) if len(sys.argv) > 8 else 0.08
n_move = int(sys.argv[9]) if len(sys.argv) > 9 else 50
n_desc = int(sys.argv[10]) if len(sys.argv) > 10 else 40
d = np.array([dx, dy]) / np.hypot(dx, dy)
incl = np.degrees(np.arctan(-slope))  # slope = dz/da (negative when dropping toward the end)
top = np.array([cx + a_g * d[0], cy + a_g * d[1], cz + slope * a_g])
radius = 0.015
g = top - np.array([0, 0, radius])
rot = grasp_frame(d, incl)
f = flange_for_grasp(g, rot)
z_axis = rot.apply([0, 0, 1])
f_pre = f - pre_up * z_axis
print('incline', round(incl, 1), 'grasp pt', g.round(4), 'flange', f.round(4), 'pre', f_pre.round(4))
print('rotvec', rot.as_rotvec().round(4), 'fingers', rot.apply([1, 0, 0]).round(3), 'palm normal', z_axis.round(3))
state = json.load(open('output/state.json'))
p0, r0, h0 = state_pose(state)
near = continuous_rotvec(r0, np.array([-np.pi, 0, 0]))
rows1 = interp_rows(p0, r0, h0, f_pre, rot, HAND_OPEN, n_move, near=near)
rows1 += [rows1[-1]] * 15
rows2 = interp_rows(f_pre, rot, HAND_OPEN, f, rot, HAND_OPEN, n_desc, near=np.array(rows1[-1][3:6]))
rows2 += [rows2[-1]] * 15
json.dump(rows1, open('output/rows_pre.json', 'w'))
json.dump(rows2, open('output/rows_desc.json', 'w'))
json.dump({'grasp_pt': g.tolist(), 'flange': f.tolist(), 'flange_pre': f_pre.tolist(), 'rotvec': rows2[-1][3:6]},
          open('output/grasp_plan.json', 'w'))
print(len(rows1), len(rows2))
