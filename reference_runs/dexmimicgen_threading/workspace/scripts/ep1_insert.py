"""Insert needle along -y world by moving robot1's target 2 mm/step; robot0 holds the ring."""
import sys, json
sys.path.insert(0, '/workspace')
import numpy as np
from tlib import *
rb = Robot()
dz = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
depth = float(sys.argv[2]) if len(sys.argv) > 2 else 0.075
a1 = rb.arm(1)
R = a1['R'].copy()
R = Rot.from_euler('x', np.deg2rad(-2.45)).as_matrix() @ (Rot.from_euler('z', np.pi / 2).as_matrix() @ np.diag([1., -1., -1.]))
start = np.array([-0.1006, 0.1194, 1.0518 + dz])
h0 = hold_target(rb.arm(0), GRIP_CLOSE)
t = dict(pos=start.copy(), R=R, grip=GRIP_CLOSE)
n = move(rb, [h0, t], max_steps=30, tol_p=0.0005, tol_r=0.004, ki=0.1)
print('aligned', n, np.round(rb.arm(1)['pos'], 4))
integ = np.zeros(3)
for k in range(200):
    off = min(depth, 0.002 * (k + 1))
    t['pos'] = start + [0, -off, 0]
    cur = rb.arm(1)
    a = make_action(cur, t, 1.0, 1.0, 0.5, 0.5)
    integ = np.clip(integ + (t['pos'] - cur['pos']) * [1, 0, 1], -0.2, 0.2)
    a[:3] += 0.1 * integ / POS_SCALE
    rb.step(make_action(rb.arm(0), h0), a)
    if k % 5 == 0 or rb.success:
        p = rb.arm(1)['pos']
        print(rb.step_count, 'off', round(off, 3), 'p1', np.round(p, 4), 'lag', np.round(t['pos'] - p, 4), 'succ', rb.success)
    if rb.success or rb.done:
        break
    if off >= depth and np.linalg.norm(t['pos'] - rb.arm(1)['pos']) < 0.001:
        break
print('final', rb.step_count, np.round(rb.arm(1)['pos'], 4), 'succ', rb.success, 'done', rb.done)
cams = rb.cameras()
np.savez_compressed('/workspace/data/insert1.npz', **cams)
json.dump(rb.obs, open('/workspace/data/insert1_obs.json', 'w'))
