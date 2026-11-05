"""Ep1 experiment: robot1 top-down grasp of the needle cube (fingers across the shaft), lift, grab cameras."""
import sys, json
sys.path.insert(0, '/workspace')
import numpy as np
from tlib import *

rb = Robot()
cube_c = np.array([-0.1386, 0.1677, 0.8198])
yaw = np.deg2rad(8.3)
Rg = Rot.from_euler('z', yaw).as_matrix() @ np.diag([1., -1., -1.])
h0 = hold_target(rb.arm(0), GRIP_OPEN)

def log(msg):
    a1 = rb.arm(1)
    print(rb.step_count, msg, 'p1', np.round(a1['pos'], 4), 'g1', np.round(a1['grip'], 4))

t = dict(pos=cube_c + [0, 0, 0.08], R=Rg, grip=GRIP_OPEN)
n = move(rb, [h0, t], max_steps=80, tol_p=0.003, tol_r=0.02); log(f'above ({n})')
t = dict(pos=cube_c + [0, 0, 0.002], R=Rg, grip=GRIP_OPEN, vmax=0.5)
n = move(rb, [h0, t], max_steps=60, tol_p=0.0015, tol_r=0.015, ki=0.1); log(f'down ({n})')
t['grip'] = GRIP_CLOSE
for k in range(15):
    rb.step(make_action(rb.arm(0), h0, 1, 1), make_action(rb.arm(1), t, 1, 1))
log('closed')
t = dict(pos=cube_c + [0, 0, 0.13], R=Rg, grip=GRIP_CLOSE, vmax=0.5)
n = move(rb, [h0, t], max_steps=80, tol_p=0.002, tol_r=0.015, ki=0.1); log(f'lifted ({n})')
for k in range(10):
    rb.step(make_action(rb.arm(0), h0, 1, 1), make_action(rb.arm(1), t, 1, 1))
log('settled')
cams = rb.cameras()
np.savez_compressed('/workspace/data/needle_lift.npz', **cams)
json.dump(rb.obs, open('/workspace/data/needle_lift_obs.json', 'w'))
