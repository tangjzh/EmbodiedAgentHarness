import sys, json, os
sys.path.insert(0, '/workspace')
import numpy as np
import robot_api as robot

res = robot.step([0.0]*14, repeat=1)
print('step keys', list(res.keys()))
print({k: v for k, v in res.items() if k != 'observation'})
sys.path.insert(0, '/workspace')
from tlib import Robot
R = Robot()
def a_(dpos=(0,0,0), drot=(0,0,0), g=-1):
    return list(dpos) + list(drot) + [g]
rec = []
def rec_state(tag):
    r0, r1 = R.arm(0), R.arm(1)
    rec.append(dict(tag=tag, t=R.step_count, p0=r0['pos'].tolist(), q0=r0['quat'].tolist(),
                    p1=r1['pos'].tolist(), q1=r1['quat'].tolist(), g0=r0['grip'].tolist(), g1=r1['grip'].tolist()))
rec_state('start')
for k in range(8):
    R.step(a_((0,0,0.4)), a_((0,0,0.4))); rec_state('up')
for k in range(8):
    R.step(a_(), a_()); rec_state('rest1')
for k in range(8):
    R.step(a_((0.4,0,0)), a_((0,-0.4,0))); rec_state('x0_ym1')
for k in range(8):
    R.step(a_(), a_()); rec_state('rest2')
for k in range(4):
    R.step(a_(drot=(0,0,0.4)), a_(drot=(0.3,0,0))); rec_state('rot')
for k in range(8):
    R.step(a_(), a_()); rec_state('rest3')
json.dump(rec, open('/workspace/data/calib1.json','w'))
for r in rec:
    print(r['tag'], r['t'], np.round(r['p0'],4), np.round(r['p1'],4), np.round(r['q0'],3), np.round(r['q1'],3))
