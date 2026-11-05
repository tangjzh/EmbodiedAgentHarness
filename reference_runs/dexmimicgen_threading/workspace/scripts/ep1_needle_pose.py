"""Ep1 experiment: robot1 brings needle (shaft along -hand x) to a pose with hand x = +y world; robot0 holds tripod."""
import sys, json
sys.path.insert(0, '/workspace')
import numpy as np
from tlib import *

rb = Robot()
pos = np.array(json.loads(sys.argv[1]))
th = np.deg2rad(float(sys.argv[2]))
tag = sys.argv[3]
vmax = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
R = Rot.from_euler('z', th).as_matrix() @ np.diag([1., -1., -1.])
h0 = hold_target(rb.arm(0), GRIP_CLOSE)
t = dict(pos=pos, R=R, grip=GRIP_CLOSE, vmax=vmax, wmax=0.4)
n = move(rb, [h0, t], max_steps=120, tol_p=0.0015, tol_r=0.01, ki=0.1)
for k in range(10):
    rb.step(make_action(rb.arm(0), h0), make_action(rb.arm(1), t))
a1 = rb.arm(1)
print(rb.step_count, f'moved ({n})', 'p1', np.round(a1['pos'], 4), 'err', np.round(errors(a1, t), 4),
      'p0', np.round(rb.arm(0)['pos'], 4), 'succ', rb.success)
cams = rb.cameras()
np.savez_compressed(f'/workspace/data/{tag}.npz', **cams)
json.dump(rb.obs, open(f'/workspace/data/{tag}_obs.json', 'w'))
