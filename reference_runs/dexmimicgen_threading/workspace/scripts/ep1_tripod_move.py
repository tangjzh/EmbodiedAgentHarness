"""Ep1 experiment: lift the tripod by the ring with robot0, then rotate/move to a canonical pose."""
import sys, json
sys.path.insert(0, '/workspace')
import numpy as np
from tlib import *

stage = sys.argv[1]
rb = Robot()
h1 = hold_target(rb.arm(1), GRIP_CLOSE)

def log(msg):
    a0 = rb.arm(0)
    print(rb.step_count, msg, 'p0', np.round(a0['pos'], 4), 'g0', np.round(a0['grip'], 4))

def grab(tag):
    cams = rb.cameras()
    np.savez_compressed(f'/workspace/data/{tag}.npz', **cams)
    json.dump(rb.obs, open(f'/workspace/data/{tag}_obs.json', 'w'))

a0 = rb.arm(0)
if stage == 'lift':
    t = dict(pos=a0['pos'] + [0, 0, 0.10], R=a0['R'], grip=GRIP_CLOSE, vmax=0.3)
    n = move(rb, [t, h1], max_steps=80, tol_p=0.002, tol_r=0.01, ki=0.1); log(f'lift ({n})')
    for k in range(10):
        rb.step(make_action(rb.arm(0), t), make_action(rb.arm(1), h1))
    log('settled')
    grab('tripod_lift')
elif stage == 'canon':
    pos = np.array(json.loads(sys.argv[2])); th = np.deg2rad(float(sys.argv[3]))
    R = Rot.from_euler('z', th).as_matrix() @ np.diag([1., -1., -1.])
    t = dict(pos=pos, R=R, grip=GRIP_CLOSE, vmax=0.3, wmax=0.4)
    n = move(rb, [t, h1], max_steps=120, tol_p=0.002, tol_r=0.01, ki=0.1); log(f'canon ({n})')
    for k in range(15):
        rb.step(make_action(rb.arm(0), t), make_action(rb.arm(1), h1))
    log('settled')
    grab('tripod_canon')
