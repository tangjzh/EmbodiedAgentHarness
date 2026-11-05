"""Shared helpers: kinematics from public model.xml, geometry, logging wrapper."""
import os, json, time
import numpy as np

B = np.array([-0.2, 0.4])   # proximal joint (x,z)
H = np.array([0.2, 0.4])    # spinner hinge (x,z)
L1 = 0.18
S_TIP = 0.1455              # fingertip capsule centre along distal link
S_A, S_B = 0.13, 0.161      # fingertip capsule axis segment along distal
R_TIP = 0.03
R_CAP = 0.04
WS = os.path.dirname(os.path.abspath(__file__))


def unpack(obs):
    pos = np.asarray(obs['position'], float)
    vel = np.asarray(obs['velocity'], float)
    touch = np.asarray(obs['touch'], float)
    q = pos[:2].copy()
    th = float(np.arctan2(-pos[2], pos[3]))
    return q, vel[:2].copy(), th, float(vel[2]), touch


def fk(q, s=S_TIP):
    q1, q2 = q
    D = B + L1 * np.array([np.cos(q1), np.sin(q1)])
    a2 = q1 + q2
    return D + s * np.array([np.cos(a2), np.sin(a2)]), D


def jac(q, s=S_TIP):
    q1, q2 = q
    a2 = q1 + q2
    return np.array([[-L1 * np.sin(q1) - s * np.sin(a2), -s * np.sin(a2)],
                     [L1 * np.cos(q1) + s * np.cos(a2), s * np.cos(a2)]])


def ik(p, elbow=+1, s=S_TIP):
    d = np.asarray(p, float) - B
    r2 = d @ d
    c2 = np.clip((r2 - L1 ** 2 - s ** 2) / (2 * L1 * s), -1, 1)
    q2 = elbow * np.arccos(c2)
    q1 = np.arctan2(d[1], d[0]) - np.arctan2(s * np.sin(q2), L1 + s * np.cos(q2))
    return np.array([q1, q2])


def seg_seg_dist(p1, q1, p2, q2):
    """min distance between segments p1-q1 and p2-q2 (2D)."""
    d1 = q1 - p1; d2 = q2 - p2; r = p1 - p2
    a = d1 @ d1; e = d2 @ d2; f = d2 @ r
    c = d1 @ r; b = d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
    t = (b * s + f) / e
    if t < 0:
        t = 0.0; s = np.clip(-c / a, 0, 1)
    elif t > 1:
        t = 1.0; s = np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def spinner_segments(th):
    u = np.array([-np.sin(th), np.cos(th)])   # body z (tip direction)
    v = np.array([np.cos(th), np.sin(th)])    # body x
    segs = []
    for sx in (+0.02, -0.02):
        c = H + sx * v
        segs.append((c - 0.09 * u, c + 0.09 * u))
    return segs


def gap(q, th):
    """surface gap between fingertip capsule and spinner (negative = penetration)."""
    D = B + L1 * np.array([np.cos(q[0]), np.sin(q[0])])
    a2 = q[0] + q[1]
    u2 = np.array([np.cos(a2), np.sin(a2)])
    pa, pb = D + S_A * u2, D + S_B * u2
    return min(seg_seg_dist(pa, pb, s0, s1) for s0, s1 in spinner_segments(th)) - R_TIP - R_CAP


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Rec:
    """Wraps robot.step, records everything to memory and saves npz."""

    def __init__(self, robot, name):
        self.robot = robot; self.name = name
        self.rows = []
        o = robot.observe()
        self.last = o
        self.t0 = time.time()

    @staticmethod
    def get_obs(r):
        if isinstance(r, dict):
            if 'observation' in r:
                return r['observation'], r
            return r, r
        if isinstance(r, (tuple, list)):
            return r[0], r
        return r, r

    def obs(self):
        o = self.last
        return o['observation'] if 'observation' in o else o

    def step(self, a, tag=0):
        a = np.clip(np.asarray(a, float), -1, 1)
        r = self.robot.step([float(a[0]), float(a[1])], 1)
        o, full = self.get_obs(r)
        self.last = full if isinstance(full, dict) else {'observation': o}
        q, dq, th, w, touch = unpack(o)
        rew = None
        if isinstance(full, dict):
            for k in ('reward', 'rewards', 'native_reward'):
                if k in full:
                    rew = full[k]; break
        if isinstance(rew, (list, tuple)):
            rew = rew[-1] if len(rew) else None
        self.rows.append(np.r_[q, dq, th, w, touch, a, (np.nan if rew is None else float(rew)), tag])
        term = bool(full.get('terminal', False)) if isinstance(full, dict) else False
        return o, full, term

    def save(self):
        os.makedirs(os.path.join(WS, 'logs'), exist_ok=True)
        arr = np.array(self.rows)
        np.save(os.path.join(WS, 'logs', self.name + '.npy'), arr)
        return arr
# columns: q1 q2 dq1 dq2 th w touch0 touch1 a0 a1 rew tag
