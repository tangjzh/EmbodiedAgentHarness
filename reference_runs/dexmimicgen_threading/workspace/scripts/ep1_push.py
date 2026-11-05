"""Push the needle further through the ring along -y (1 mm/step target) until success, a y limit, or robot0 disturbance."""
import sys, json
sys.path.insert(0, '/workspace')
import numpy as np
from tlib import *
rb = Robot()
ylim = float(sys.argv[1]); rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001
h0 = hold_target(rb.arm(0), GRIP_CLOSE)
p0_ref = rb.arm(0)['pos'].copy()
a1 = rb.arm(1); R = a1['R'].copy()
x0, z0 = -0.1006, 1.0527
ty = a1['pos'][1]
integ = np.zeros(3)
for k in range(200):
    ty = max(ylim, ty - rate)
    t = dict(pos=np.array([x0, ty, z0]), R=R, grip=GRIP_CLOSE)
    cur = rb.arm(1)
    a = make_action(cur, t, 1.0, 1.0, 0.3, 0.3)
    e = t['pos'] - cur['pos']; e[1] = 0
    integ = np.clip(integ + e, -0.2, 0.2)
    a[:3] = np.clip(a[:3] + 0.1 * integ / POS_SCALE, -1, 1)
    rb.step(make_action(rb.arm(0), h0, 1.0, 1.0), a)
    d0 = np.linalg.norm(rb.arm(0)['pos'] - p0_ref)
    if k % 2 == 0 or rb.success:
        print(rb.step_count, 'p1', np.round(rb.arm(1)['pos'], 4), 'ty', round(ty, 4), 'd0', round(d0 * 1000, 2), 'mm', 'succ', rb.success)
    if rb.success or rb.done or d0 > 0.002:
        break
    if ty <= ylim and abs(rb.arm(1)['pos'][1] - ylim) < 0.0008:
        break
print('final', rb.step_count, 'p1', np.round(rb.arm(1)['pos'], 4), 'p0', np.round(rb.arm(0)['pos'], 4), 'succ', rb.success)
cams = rb.cameras()
np.savez_compressed('/workspace/data/push.npz', **cams)
json.dump(rb.obs, open('/workspace/data/push_obs.json', 'w'))
