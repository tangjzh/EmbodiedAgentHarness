"""Move arms to look-at poses given on argv as JSON and grab cameras.
argv[1]: tag; argv[2]: JSON list of 2 entries: null (hold) or [campos(3), target(3), up_hint(3)]"""
import sys, json
sys.path.insert(0, '/workspace')
import numpy as np
from tlib import *

tag = sys.argv[1]
spec = json.loads(sys.argv[2]) if len(sys.argv) > 2 else [None, None]
grip = json.loads(sys.argv[3]) if len(sys.argv) > 3 else [-1, -1]
rb = Robot()
tgts = []
for i in range(2):
    a = rb.arm(i)
    if spec[i] is None:
        tgts.append(hold_target(a, grip[i]))
    else:
        cp, tp, up = [np.array(v, float) for v in spec[i]]
        ph, Rh = look_at_pose(cp, tp, up)
        tgts.append(dict(pos=ph, R=Rh, grip=grip[i]))
n = move(rb, tgts, max_steps=80, tol_p=0.002, tol_r=0.02, verbose=False)
n2 = move(rb, tgts, max_steps=15, tol_p=0.001, tol_r=0.01, kp=1.0, kr=1.0)
print('steps', n, n2, 'now', rb.step_count)
for i in range(2):
    print(i, 'err', np.round(errors(rb.arm(i), tgts[i]), 4), 'pos', np.round(rb.arm(i)['pos'], 4))
cams = rb.cameras()
np.savez_compressed(f'/workspace/data/{tag}.npz', **cams)
json.dump(rb.obs, open(f'/workspace/data/{tag}_obs.json', 'w'))
