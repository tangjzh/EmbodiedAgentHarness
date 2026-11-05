"""Generate rows from the start pose to a downward-looking camera pose; print JSON for preview."""
import sys, json
sys.path.insert(0, 'experiments')
import numpy as np
from scipy.spatial.transform import Rotation as R
from motion import *

state = [-0.44294797221632903, 0, 1.5170134972986655, 4.36497393134852e-11, -0.999996875001627,
         1.7459859336099726e-08, -0.0024999973958342954] + HAND_OPEN
p0, r0, h0 = state_pose(state)
tx, ty, tz, pitch, yaw, n = [float(a) for a in sys.argv[1:7]]
r1 = R.from_euler('z', np.radians(yaw)) * R.from_euler('y', np.radians(pitch)) * R.from_rotvec([np.pi, 0, 0])
rows = interp_rows(p0, r0, h0, np.array([tx, ty, tz]), r1, HAND_OPEN, int(n), near=np.array([-np.pi, 0, 0]))
rows += [rows[-1]] * 20
json.dump(rows, open('output/rows.json', 'w'))
print(len(rows), rows[-1][:6])
