"""Offline unit checks for motion helpers and controller geometry."""
import os
import sys
import math

os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.insert(0, 'code/hammer')
import numpy as np
from scipy.spatial.transform import Rotation as R
import motion as M
import controller as C

print('palm_down(0) rotvec', M.palm_down(0).as_rotvec().round(4))
r = M.grasp_rotation([-0.8397, -0.543], math.radians(15.5))
print('E1 grasp rotvec', M.continuous_rotvec(r, np.array([-np.pi, 0, 0])).round(4), 'expected [-2.545, 1.385, -0.189]')
print('E1 grasp yaw deg', round(math.degrees(M.yaw_of(r)), 2), 'expected -57.1')
# E1 grasp flange
g = np.array([-0.2531 + 0.015 * -0.8397, -0.4798 + 0.015 * -0.543, 1.0502 - 0.278 * 0.015 - 0.015])
F = g - r.apply(C.GRASP_OFFSET_F)
print('E1 flange', F.round(4), 'expected [-0.3242, -0.4246, 1.084]')
# wrist camera calibration check vs E1 step 0 camera pose
st = [-0.44294797221632903, 0, 1.5170134972986655, 4.36497393134852e-11, -0.999996875001627, 1.7459859336099726e-08, -0.0024999973958342954]
rf = M.rot_from_state(st + [0] * 16)
cam = np.array(st[:3]) + rf.apply(C.CAM_OFF_F)
print('wrist cam pos', cam.round(4), 'expected [-0.5827, 0.03, 1.4763]')
q = [0.6117026911951683, 0.3542081838182911, -0.35520854702233023, -0.611705213499951]
rc = R.from_quat([q[1], q[2], q[3], q[0]])
look = rc.apply([0, 0, -1])
print('wrist look world', look.round(4), 'from calib', rf.apply(C.CAM_DIR_F).round(4))
# interp rows sanity
rows = M.interp_rows(np.zeros(3), M.palm_down(0), M.HAND_OPEN, np.array([0.1, 0, 0]), M.palm_down(1.0), M.HAND_OPEN, 5)
print('rows', len(rows), [round(v, 3) for v in rows[-1][:6]])
print('move_steps 0.5 m', M.move_steps(np.zeros(3), M.palm_down(0), np.array([0.5, 0, 0]), M.palm_down(0)))
