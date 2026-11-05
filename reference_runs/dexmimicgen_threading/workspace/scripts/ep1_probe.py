"""Probe success criterion: apply small offsets to robot1 (world) from current pose, hold, report success."""
import sys, json
sys.path.insert(0, '/workspace')
import numpy as np
from tlib import *
rb = Robot()
offs = json.loads(sys.argv[1])
hold = int(sys.argv[2]) if len(sys.argv) > 2 else 6
h0 = hold_target(rb.arm(0), GRIP_CLOSE)
base = rb.arm(1)['pos'].copy(); R = rb.arm(1)['R'].copy()
for off in offs:
    t = dict(pos=base + np.array(off), R=R, grip=GRIP_CLOSE)
    for k in range(hold):
        rb.step(make_action(rb.arm(0), h0), make_action(rb.arm(1), t, 1.0, 1.0, 0.3, 0.3))
        if rb.success or rb.done:
            break
    print(rb.step_count, 'off', off, 'p1', np.round(rb.arm(1)['pos'], 4), 'p0', np.round(rb.arm(0)['pos'], 4), 'succ', rb.success)
    if rb.success or rb.done:
        break
cams = rb.cameras()
np.savez_compressed('/workspace/data/probe.npz', **cams)
json.dump(rb.obs, open('/workspace/data/probe_obs.json', 'w'))
