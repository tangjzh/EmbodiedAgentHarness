"""TCP pose servo helpers for pd_ee_delta_pose.

Empirical (ep1): translation acts in world axes; per-step displacement v_k ~= 0.24*u_k + 0.40*v_{k-1}
(u in metres, saturates at 0.1).  Rotation acts about world axes at the TCP but with NEGATIVE sign:
world rotation per step ~= -0.39 * (action_rot * 0.1).
"""
import numpy as np
from geom import quat_wxyz_to_R, rotvec

BASE_XY = np.array([-0.615, 0.0])   # robot base (world)
REACH_MARGIN = 0.02

POS_SCALE = 0.1   # metres per unit action
ROT_SCALE = 0.1   # radians per unit action
ROT_SIGN = -1.0   # measured: commanded world rotation is applied with inverted sign


def tcp_pose(obs):
    t = np.array(obs['robot']['tcp_pose'][0], dtype=np.float64)
    return t[:3], quat_wxyz_to_R(t[3:])


def gripper_width(obs):
    q = obs['robot']['agent']['qpos'][0]
    return q[7] + q[8]


def rho_max(z):
    """Horizontal reach (from the base axis) of a top-down TCP vs height; fitted to Panda IK (kin.py, ep5)."""
    return 0.806 - 0.65 * (z - 0.12) ** 2


def reach_excess(p, margin=REACH_MARGIN):
    return float(np.hypot(p[0] - BASE_XY[0], p[1] - BASE_XY[1]) - (rho_max(p[2]) - margin))


def clip_reach(p, d, margin=REACH_MARGIN):
    """Shrink the commanded translation d so the controller target p+d stays IK-feasible.  (With
    use_target=False the native IK solves for p + delta every step; an infeasible target makes the IK fail
    and the arm freezes in place -- ep5.)"""
    if reach_excess(p + d, margin) <= 0:
        return d
    if reach_excess(p, margin) > 0:
        # already at the shell: drop the outward radial component
        u = np.array([p[0] - BASE_XY[0], p[1] - BASE_XY[1], 0.0])
        u /= np.linalg.norm(u) + 1e-12
        d = d - max(float(d @ u), 0.0) * u
        if p[2] + d[2] > 0.12 and d[2] > 0:
            d[2] = 0.0
        return d
    lo, hi = 0.0, 1.0
    for _ in range(14):
        m = 0.5 * (lo + hi)
        if reach_excess(p + m * d, margin) <= 0:
            lo = m
        else:
            hi = m
    return lo * d


def topdown_R(yaw, tilt=0.0):
    """TCP rotation: approach (z) pointing down, x along yaw; optional tilt about TCP y (pitch)."""
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    R = np.stack([x, y, z], 1)
    if tilt != 0.0:
        c, s = np.cos(tilt), np.sin(tilt)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        R = R @ Ry
    return R


class Servo:
    """Momentum-aware proportional pose servo.

    Translation damping is velocity-scheduled: kd_eff = kd * min(1, |v|/v0) (v = last per-step displacement).
    Full damping while the arm moves fast removes the ~2 cm braking overshoot of the kp=2 servo (per-step model
    fitted on logs: v_k ~ 0.24 u_k + 0.41 v_{k-1}); near rest it fades out so small limit cycles (seen in ep1
    with constant kd) are not excited.  Rotation gets the same scheduled damping (krd, w0): a saturated yaw turn
    (~0.037 rad/step) otherwise overshoots ~0.02 rad at the end of staging (ep8)."""

    def __init__(self, kp=2.0, kd=1.5, kr=2.0, max_p=1.0, max_r=1.0, v0=0.02, krd=1.5, w0=0.02):
        self.kp, self.kd, self.kr, self.max_p, self.max_r, self.v0 = kp, kd, kr, max_p, max_r, v0
        self.krd, self.w0 = krd, w0
        self.prev_p = None
        self.prev_R = None

    def reset(self):
        self.prev_p = None
        self.prev_R = None

    def action(self, obs, p_t, R_t, grip, kp=None, max_p=None, max_r=None, kd=None, ff=None):
        """ff: feedforward translation (metres per step) added to the P-D command, e.g. payload sag compensation."""
        p, R = tcp_pose(obs)
        kp = self.kp if kp is None else kp
        kd = self.kd if kd is None else kd
        max_p = self.max_p if max_p is None else max_p
        max_r = self.max_r if max_r is None else max_r
        v = np.zeros(3) if self.prev_p is None else p - self.prev_p
        w = np.zeros(3) if self.prev_R is None else rotvec(R @ self.prev_R.T)
        self.prev_p = p.copy()
        self.prev_R = R.copy()
        ep = p_t - p
        er = rotvec(R_t @ R.T)
        u = kp * ep - kd * min(1.0, np.linalg.norm(v) / self.v0) * v
        if ff is not None:
            u = u + ff
        a = np.zeros(7)
        a[:3] = u / POS_SCALE
        m = np.max(np.abs(a[:3]))
        if m > max_p:
            a[:3] *= max_p / m      # keep direction when saturating
        a[:3] = clip_reach(p, a[:3] * POS_SCALE) / POS_SCALE
        ur = self.kr * er - self.krd * min(1.0, np.linalg.norm(w) / self.w0) * w
        ar = ROT_SIGN * ur / ROT_SCALE
        n = np.linalg.norm(ar)
        if n > max_r:
            ar = ar * max_r / n
        a[3:6] = ar
        a[6] = grip
        return a, ep, er
