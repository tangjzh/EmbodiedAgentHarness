import numpy as np
from scipy.spatial.transform import Rotation as R
import json, sys

R0 = R.from_rotvec([np.pi, 0, 0])
pitch = np.radians(float(sys.argv[1]) if len(sys.argv) > 1 else 60)
yaw = np.radians(float(sys.argv[2]) if len(sys.argv) > 2 else 0)
Rr = R.from_euler('z', yaw) * R.from_euler('y', pitch) * R0
cam_off = np.array([-0.14, -0.03, 0.04])
cam_dir_f = np.array([0.8654, 0.0012, 0.5011])
print('rotvec', Rr.as_rotvec().round(5).tolist())
print('cam offset world', Rr.apply(cam_off).round(4), 'cam dir world', Rr.apply(cam_dir_f).round(4))
print('finger dir (flange x) world', Rr.apply([1, 0, 0]).round(4), 'flange z world', Rr.apply([0, 0, 1]).round(4))
