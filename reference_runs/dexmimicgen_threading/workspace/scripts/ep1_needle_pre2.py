"""Pre-insertion with corrected pitch (tip up 2.45 deg) and height."""
import sys, json
sys.path.insert(0, '/workspace')
import numpy as np
from tlib import *
rb = Robot()
R = Rot.from_euler('x', np.deg2rad(-2.45)).as_matrix() @ (Rot.from_euler('z', np.pi / 2).as_matrix() @ np.diag([1., -1., -1.]))
pos = np.array([-0.1005, 0.1195, 1.0519])
h0 = hold_target(rb.arm(0), GRIP_CLOSE)
t = dict(pos=pos, R=R, grip=GRIP_CLOSE, vmax=0.3, wmax=0.3)
n = move(rb, [h0, t], max_steps=100, tol_p=0.001, tol_r=0.005, ki=0.1)
for k in range(10):
    rb.step(make_action(rb.arm(0), h0), make_action(rb.arm(1), t))
a1 = rb.arm(1)
print(rb.step_count, f'moved ({n})', 'p1', np.round(a1['pos'], 4), 'err', np.round(errors(a1, t), 4), 'p0', np.round(rb.arm(0)['pos'], 4), 'succ', rb.success)
cams = rb.cameras()
np.savez_compressed('/workspace/data/needle_pre2.npz', **cams)
json.dump(rb.obs, open('/workspace/data/needle_pre2_obs.json', 'w'))
