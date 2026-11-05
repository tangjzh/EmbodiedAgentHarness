# Active-view test: release peg, move hand camera to view the hole at ~32 deg off-axis, save frames.
import sys, json
import numpy as np
sys.path.insert(0, '/workspace')
import robot_api as robot
from motion import *
from geom import *
from perception import detect_hole

xb = np.array([0.2114, 0.9774, 0.0]); yaw = np.arctan2(xb[1], xb[0])
hole = np.array([0.0175, 0.1113, 0.1208])
C_OFF = np.array([0.0465, -0.02, -0.0674])
obs = robot.observe()
p0, R0 = tcp_pose(obs)
R_t = topdown_R(yaw)
sv = Servo(kp=2.0, kd=0.0, kr=2.0)
cam_t = hole - 0.075 * xb + np.array([0, 0, 0.12])
T = cam_t - R_t @ C_OFF
print('TCP target', np.round(T, 4))
k = 0
def save(tag):
    cams = robot.cameras()
    np.savez_compressed(f'/workspace/data/{tag}.npz', **cams)
    json.dump(obs, open(f'/workspace/data/{tag}_obs.json', 'w'))
    return cams
# open gripper holding pose
for i in range(2):
    a, ep, er = sv.action(obs, p0, R_t, 1.0)
    res = robot.step(a.tolist()); obs = res['observation']
print('opened', obs['native_step'], gripper_width(obs))
save('ep1_d_open')
for i in range(10):
    a, ep, er = sv.action(obs, T, R_t, 1.0)
    res = robot.step(a.tolist()); obs = res['observation']
    p, R = tcp_pose(obs)
    e = np.linalg.norm(T - p)
    cams = save(f'ep1_d_v{i}')
    r = detect_hole(obs, cams, 'hand_camera')
    s = 'none' if r is None or r.get('hole') is None else 'hole %s u %s z %s npx %d %s' % (
        np.round(r['hole'], 4), np.round(r['u_ext'], 4), np.round(r['z_ext'], 4), r['npx'], {k_: v for k_, v in r['touches'].items() if v})
    print(obs['native_step'], 'err %.4f' % e, s)
    if e < 0.004 and i >= 3:
        break
