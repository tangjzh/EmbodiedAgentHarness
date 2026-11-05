"""Flange target that puts a flange-frame point at a world target, with given yaw/roll (palm-down base).

usage: face_target.py sx sy sz tx ty tz yaw roll
"""
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

sx, sy, sz, tx, ty, tz, yaw, roll = [float(v) for v in sys.argv[1:9]]
r = R.from_euler('z', np.radians(yaw)) * R.from_rotvec([np.pi, 0, 0]) * R.from_euler('x', np.radians(roll))
f = np.array([tx, ty, tz]) - r.apply([sx, sy, sz])
print('flange target', f.round(4).tolist())
