import sys, json
import numpy as np
sys.path.insert(0, '/workspace')
import robot_api as robot
from motion import *
from geom import *

c = np.array([0.0533, -0.2302]); yaw = 1.750; L = 0.111; r = 0.023
ax = np.array([np.cos(yaw), np.sin(yaw)])
gxy = c - (0.5 * L + 0.01) * ax
R_t = topdown_R(yaw)
obs = robot.observe()
sv = Servo()
log = []

def run(p_t, grip, nmax, tol_p, tol_r, label):
    global obs
    for i in range(nmax):
        a, ep, er = sv.action(obs, p_t, R_t, grip)
        if np.linalg.norm(ep) < tol_p and np.linalg.norm(er) < tol_r:
            print(label, 'converged before step', i)
            return
        res = robot.step(a.tolist())
        obs = res['observation']
        p2, R2 = tcp_pose(obs)
        log.append(dict(label=label, a=a.tolist(), ep=ep.tolist(), er=er.tolist(), p=p2.tolist(), R=R2.tolist(),
                        qpos=obs['robot']['agent']['qpos'][0], step=obs['native_step']))
        print(label, obs['native_step'], 'ep %.4f er %.4f' % (np.linalg.norm(ep), np.linalg.norm(er)),
              'a', np.round(a, 3), 'grip w %.4f' % gripper_width(obs))

run(np.array([gxy[0], gxy[1], 0.07]), 1.0, 10, 0.003, 0.02, 'pre')
run(np.array([gxy[0], gxy[1], 0.0235]), 1.0, 8, 0.0015, 0.02, 'down')
for i in range(5):
    res = robot.step([0, 0, 0, 0, 0, 0, -1.0])
    obs = res['observation']
    p2, R2 = tcp_pose(obs)
    print('close', obs['native_step'], 'grip w %.4f' % gripper_width(obs), 'q', np.round(obs['robot']['agent']['qpos'][0][7:], 4), 'tcp', np.round(p2, 4))
    log.append(dict(label='close', p=p2.tolist(), R=R2.tolist(), qpos=obs['robot']['agent']['qpos'][0], step=obs['native_step']))
json.dump(log, open('/workspace/data/ep1_b_log.json', 'w'))
cams = robot.cameras()
np.savez_compressed('/workspace/data/ep1_grasped.npz', **cams)
json.dump(obs, open('/workspace/data/ep1_grasped_obs.json', 'w'))
