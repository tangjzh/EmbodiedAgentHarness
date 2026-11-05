"""Shared helpers for the TwoArmThreading controller: state parsing, logging, stepping."""
import json
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import robot_api as robot

POS_SCALE = 0.05
ROT_SCALE = 0.5
LOG_PATH = os.environ.get('TLIB_LOG')   # dev diagnostics only; no step log unless set


def arm_state(obs, i):
    r = obs['robot']
    p = f'robot{i}_'
    q = np.array(r[p + 'eef_quat'])
    return dict(pos=np.array(r[p + 'eef_pos']), quat=q, R=Rot.from_quat(q).as_matrix(),
                grip=np.array(r[p + 'gripper_qpos']), jpos=np.array(r[p + 'joint_pos']))


class Robot:
    """Thin wrapper that tracks the latest observation and logs every step."""

    def __init__(self, log=True):
        self.obs = robot.observe()
        self.log = log
        self.done = False
        self.success = False

    @property
    def step_count(self):
        return self.obs['native_step']

    def arm(self, i):
        return arm_state(self.obs, i)

    def step(self, a0, a1, repeat=1):
        a = np.concatenate([np.asarray(a0, float), np.asarray(a1, float)])
        a = np.clip(a, -1, 1)
        res = robot.step(a.tolist(), repeat=repeat)
        self.obs = res['observation'] if 'observation' in res else robot.observe()
        self.success = bool(self.obs.get('native_success', False) or res.get('native_success', False))
        self.done = bool(self.obs.get('terminal', False) or res.get('terminal', False))
        if self.log and LOG_PATH:
            try:
                r0, r1 = self.arm(0), self.arm(1)
                with open(LOG_PATH, 'a') as f:
                    f.write(json.dumps(dict(t=self.step_count, a=a.tolist(),
                                            p0=r0['pos'].tolist(), q0=r0['quat'].tolist(), g0=r0['grip'].tolist(),
                                            p1=r1['pos'].tolist(), q1=r1['quat'].tolist(), g1=r1['grip'].tolist(),
                                            succ=self.success)) + '\n')
            except Exception:  # read-only workspace etc.: logging must never break the episode
                self.log = False
        return res

    def cameras(self):
        return robot.cameras()


def rotvec_err(R_target, R_cur):
    """World-frame rotation vector taking R_cur to R_target."""
    return Rot.from_matrix(R_target @ R_cur.T).as_rotvec()


# ---------------------------------------------------------------- motion
# Wrist camera pose in the hand (eef_quat) frame, measured from calibration.
CAM_IN_HAND = np.array([[0., -1., 0., 0.05],
                        [1., 0., 0., 0.],
                        [0., 0., 1., -0.097],
                        [0., 0., 0., 1.]])
GRIP_OPEN, GRIP_CLOSE = -1.0, 1.0


def make_action(cur, tgt, kp=1.5, kr=1.5, vmax=1.0, wmax=1.0):
    """Delta action (7,) driving arm state `cur` toward target dict(pos, R, grip)."""
    a = np.zeros(7)
    dp = tgt['pos'] - cur['pos']
    a[:3] = kp * dp / POS_SCALE
    n = np.linalg.norm(a[:3])
    if n > vmax:
        a[:3] *= vmax / n
    if tgt.get('R') is not None:
        dr = rotvec_err(tgt['R'], cur['R'])
        a[3:6] = kr * dr / ROT_SCALE
        n = np.linalg.norm(a[3:6])
        if n > wmax:
            a[3:6] *= wmax / n
    a[6] = tgt.get('grip', GRIP_OPEN)
    return a


def errors(cur, tgt):
    ep = np.linalg.norm(tgt['pos'] - cur['pos'])
    er = 0.0 if tgt.get('R') is None else np.linalg.norm(rotvec_err(tgt['R'], cur['R']))
    return ep, er


def hold_target(arm, grip):
    return dict(pos=arm['pos'].copy(), R=arm['R'].copy(), grip=grip)


def move(rb, tgts, max_steps=60, tol_p=0.003, tol_r=0.03, kp=1.0, kr=1.0, vmax=1.0, wmax=1.0,
         min_steps=0, ki=0.0, verbose=False):
    """Servo both arms to targets (list of 2 dicts); returns number of steps used.
    kp=1 puts the OSC goal exactly on the target (critically damped, no overshoot); ki>0 adds an
    integral term on the position error to cancel load sag (clipped to 2 cm of goal offset)."""
    integ = [np.zeros(3), np.zeros(3)]
    for k in range(max_steps):
        acts, done = [], True
        for i in range(2):
            cur = rb.arm(i)
            t = tgts[i]
            a = make_action(cur, t, t.get('kp', kp), kr, t.get('vmax', vmax), t.get('wmax', wmax))
            if ki > 0:
                integ[i] = np.clip(integ[i] + (t['pos'] - cur['pos']), -0.02 / ki, 0.02 / ki)
                a[:3] = np.clip(a[:3] + ki * integ[i] / POS_SCALE, -1, 1)
            acts.append(a)
            ep, er = errors(cur, t)
            if ep > t.get('tol_p', tol_p) or er > t.get('tol_r', tol_r):
                done = False
        if done and k >= min_steps:
            return k
        rb.step(acts[0], acts[1])
        if verbose:
            print(rb.step_count, [np.round(errors(rb.arm(i), tgts[i]), 4) for i in range(2)])
        if rb.done or rb.success:
            return k
    return max_steps


def look_at_pose(cam_pos, target, up_hint=np.array([1.0, 0, 0])):
    """Hand target (pos, R) so the wrist camera sits at cam_pos looking at target.
    up_hint: world direction that should appear toward the image top."""
    z = target - cam_pos
    z /= np.linalg.norm(z)
    y = -(up_hint - up_hint.dot(z) * z)  # image down = -up
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    Rc = np.stack([x, y, z], 1)
    Rh = Rc @ CAM_IN_HAND[:3, :3].T
    ph = cam_pos - Rh @ CAM_IN_HAND[:3, 3]
    return ph, Rh
