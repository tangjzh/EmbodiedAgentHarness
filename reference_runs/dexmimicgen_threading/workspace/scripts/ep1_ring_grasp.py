"""Ep1 experiment: robot0 reorients top-down above the ring (hand x along ring axis), looks, then grasps the ring."""
import sys, json
sys.path.insert(0, '/workspace')
import numpy as np
from tlib import *
from geom import deproject, cam_mats

stage = sys.argv[1] if len(sys.argv) > 1 else 'above'
rb = Robot()
a_ax = np.array([0.856, 0.516, 0.0])
c_h = np.array([-0.0097, -0.1923, 0.984])
th = np.arctan2(a_ax[1], a_ax[0])
Rt = Rot.from_euler('z', th).as_matrix() @ np.diag([1., -1., -1.])
h1 = hold_target(rb.arm(1), GRIP_CLOSE)

def log(msg):
    a0 = rb.arm(0)
    print(rb.step_count, msg, 'p0', np.round(a0['pos'], 4), 'g0', np.round(a0['grip'], 4),
          'rerr', np.round(np.linalg.norm(rotvec_err(Rt, a0['R'])), 3))

def grab(tag):
    cams = rb.cameras()
    np.savez_compressed(f'/workspace/data/{tag}.npz', **cams)
    json.dump(rb.obs, open(f'/workspace/data/{tag}_obs.json', 'w'))
    return cams

if stage == 'above':
    p = rb.arm(0)['pos']
    Rmid = Rot.from_rotvec(0.5 * rotvec_err(Rt, rb.arm(0)['R'])).as_matrix() @ rb.arm(0)['R']
    t = dict(pos=np.array([p[0] - 0.03, p[1] - 0.02, 1.08]), R=Rmid, grip=GRIP_OPEN, wmax=0.6)
    n = move(rb, [t, h1], max_steps=60, tol_p=0.01, tol_r=0.05); log(f'up ({n})')
    t = dict(pos=c_h + [0, 0, 0.10], R=Rt, grip=GRIP_OPEN, wmax=0.6)
    n = move(rb, [t, h1], max_steps=80, tol_p=0.002, tol_r=0.01, ki=0.1); log(f'above ({n})')
    grab('ring_above')
elif stage == 'grasp':
    tgt = np.array(json.loads(sys.argv[2]))
    t = dict(pos=tgt + [0, 0, 0.04], R=Rt, grip=GRIP_OPEN)
    n = move(rb, [t, h1], max_steps=60, tol_p=0.002, tol_r=0.01, ki=0.1); log(f'pre ({n})')
    t = dict(pos=tgt, R=Rt, grip=GRIP_OPEN, vmax=0.3)
    n = move(rb, [t, h1], max_steps=60, tol_p=0.0015, tol_r=0.01, ki=0.1); log(f'down ({n})')
    t['grip'] = GRIP_CLOSE
    for k in range(15):
        rb.step(make_action(rb.arm(0), t), make_action(rb.arm(1), h1))
    log('closed')
    grab('ring_grasped')
