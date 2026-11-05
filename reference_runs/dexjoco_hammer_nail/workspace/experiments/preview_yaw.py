"""Preview reachability of palm-down poses at given xy, z and yaw (robot-only simulator).

usage: preview_yaw.py x y z yaw_deg [yaw_deg ...] [--roll R]
Moves from the start pose to the target over 120 rows + 40 hold; prints the end error.
"""
import json
import math
import os
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.insert(0, '.')
sys.path.insert(0, 'code/hammer')
import numpy as np
from api.runtime import Context
from motion import HAND_OPEN, interp_rows, palm_down, rot_from_state, continuous_rotvec
from perception import decode_reply

START = [-0.44294797221632903, 0.0, 1.5170134972986655, 4.36497393134852e-11, -0.999996875001627,
         1.7459859336099726e-08, -0.0024999973958342954] + [0.0] * 12 + [0.2630000114440918, 0.0, 0.0, 0.0]
args = [a for a in sys.argv[1:]]
roll = 0.0
if '--roll' in args:
    k = args.index('--roll')
    roll = math.radians(float(args[k + 1]))
    del args[k:k + 2]
x, y, z = [float(v) for v in args[:3]]
yaws = [float(v) for v in args[3:]]
p0 = np.array(START[:3])
r0 = rot_from_state(START)
with Context() as ctx:
    for yd in yaws:
        r1 = palm_down(math.radians(yd), roll)
        rows = interp_rows(p0, r0, HAND_OPEN, np.array([x, y, z]), r1, HAND_OPEN, 110, near=np.array([-np.pi, 0, 0]))
        rows += [rows[-1]] * 40
        rep = ctx.call('dexjoco_preview_run', actions=rows, initial_state=START)
        meta, _ = decode_reply(rep)
        es = meta.get('end_state')
        if not es:
            print(yd, 'no end_state', str(meta)[:300])
            continue
        rf = rot_from_state(es)
        perr = np.linalg.norm(np.array(es[:3]) - np.array([x, y, z]))
        aerr = math.degrees((rf * r1.inv()).magnitude())
        print(f'yaw {yd:7.1f} roll {math.degrees(roll):5.1f}: pos err {perr*100:5.1f} cm, rot err {aerr:5.1f} deg, end {np.round(es[:3],3)}')
