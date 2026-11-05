import numpy as np, robot_api as robot
from lib import *

np.set_printoptions(precision=3, suppress=True, linewidth=160)
rec = Rec(robot, 'exp01_sysid')
print('obs0', rec.obs())
o, full, term = rec.step([0, 0], tag=0)
if isinstance(full, dict):
    print('step keys', list(full.keys()))
    print({k: v for k, v in full.items() if k not in ('observation_artifacts',)})
else:
    print('step type', type(full), full)

seq = []
seq += [([0, 0], 1)] * 3
seq += [([0.5, 0], 2)] * 4 + [([0, 0], 3)] * 6 + [([-0.5, 0], 4)] * 4 + [([0, 0], 5)] * 6
seq += [([0, 0.5], 6)] * 4 + [([0, 0], 7)] * 6 + [([0, -0.5], 8)] * 4 + [([0, 0], 9)] * 6
seq += [([1, 0], 10)] * 3 + [([-1, 0], 11)] * 3 + [([0, 0], 12)] * 6
seq += [([0, 1], 13)] * 3 + [([0, -1], 14)] * 3 + [([0, 0], 15)] * 6
for a, tag in seq:
    o, full, term = rec.step(a, tag)
arr = rec.save()
print('cols: q1 q2 dq1 dq2 th w t0 t1 a0 a1 rew tag')
print(arr)
print('last full', {k: v for k, v in full.items() if k not in ('observation_artifacts',)} if isinstance(full, dict) else full)
