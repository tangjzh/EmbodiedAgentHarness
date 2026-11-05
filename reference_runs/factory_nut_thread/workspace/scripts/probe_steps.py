"""Probe: a few zero-translation steps, print step reply structure and EEF/yaw response."""
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, '/workspace')
import controller as C
import robot_api as robot

obs = robot.observe()
print('start step', obs['native_step'], 'eef', obs['robot']['eef_position_world'], 'yaw', C.yaw_of(obs['robot']['eef_quaternion_wxyz']))
for i in range(6):
    t0 = time.time()
    r = robot.step([0, 0, 0, 0, 0, 1.0])
    dt = time.time() - t0
    if i == 0:
        print('reply keys', list(r.keys()))
        print({k: v for k, v in r.items() if k != 'observation'})
    o = r['observation'] if 'observation' in r else r
    print(i, 'step', o['native_step'], 'eef', np.round(o['robot']['eef_position_world'], 5), 'yaw %.4f' % C.yaw_of(o['robot']['eef_quaternion_wxyz']),
          'succ', o.get('native_success'), 'dt %.3f' % dt)
