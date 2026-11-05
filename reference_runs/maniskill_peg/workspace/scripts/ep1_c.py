import sys, json
import numpy as np
sys.path.insert(0, '/workspace')
import robot_api as robot
from motion import *
from geom import *

xb = np.array([0.2114, 0.9774, 0.0]); yaw = np.arctan2(xb[1], xb[0])
hole = np.array([0.01662, 0.11183, 0.1202])    # hole centre on entrance face
L = 0.111; head_off = 0.063 + L                    # head ahead of TCP along TCP x
dz_tcp = 0.0014                                    # TCP above peg axis
R_t = topdown_R(yaw)
obs = robot.observe()
sv = Servo(kp=2.0, kd=0.0, kr=2.0)
log = []

def tgt(standoff):
    p = hole - (head_off + standoff) * xb
    p[2] += dz_tcp
    return p

def step(a, label, extra=None):
    global obs
    res = robot.step(a.tolist())
    obs = res['observation']
    p2, R2 = tcp_pose(obs)
    e = dict(label=label, a=a.tolist(), p=p2.tolist(), R=R2.tolist(), qpos=obs['robot']['agent']['qpos'][0],
             step=obs['native_step'], succ=res['native_success'])
    if extra: e.update(extra)
    log.append(e)
    return res

# 1) lift and go to pre-insertion with standoff 3.5cm
p_pre = tgt(0.035)
for i in range(18):
    a, ep, er = sv.action(obs, p_pre, R_t, -1.0)
    if np.linalg.norm(ep) < 0.002 and np.linalg.norm(er) < 0.01:
        break
    step(a, 'pre')
    print('pre', obs['native_step'], 'ep %.4f er %.4f' % (np.linalg.norm(ep), np.linalg.norm(er)), 'grip %.4f' % gripper_width(obs))
cams = robot.cameras(); np.savez_compressed('/workspace/data/ep1_preins.npz', **cams)
json.dump(obs, open('/workspace/data/ep1_preins_obs.json', 'w'))
# 2) insert: move target along axis
s = 0.035
for i in range(28):
    s = max(s - 0.008, -0.105)
    p_t = tgt(s)
    a, ep, er = sv.action(obs, p_t, R_t, -1.0)
    res = step(a, 'ins', dict(s=s))
    p, R = tcp_pose(obs)
    along = (p - tgt(0.0)) @ xb
    lat = p - tgt(0.0) - along * xb
    print('ins', obs['native_step'], 's %.3f along %.4f lat %s' % (s, along, np.round(lat, 4)), 'er %.4f' % np.linalg.norm(er),
          'grip %.4f' % gripper_width(obs), 'succ', res['native_success'])
    if res['native_success'] or res['terminal']:
        break
json.dump(log, open('/workspace/data/ep1_c_log.json', 'w'))
cams = robot.cameras(); np.savez_compressed('/workspace/data/ep1_end.npz', **cams)
json.dump(obs, open('/workspace/data/ep1_end_obs.json', 'w'))
