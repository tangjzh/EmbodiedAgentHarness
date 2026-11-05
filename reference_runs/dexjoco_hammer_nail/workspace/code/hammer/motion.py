"""Rotations and action-row generation for the Panda + Allegro flange (world frame)."""
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

HAND_OPEN = [0.0] * 12 + [0.263, 0.0, 0.0, 0.0]
DOWN = np.array([0.0, 0.0, -1.0])


def rot_from_state(state):
    q = state[3:7]
    return R.from_quat([q[1], q[2], q[3], q[0]])


def palm_down(yaw, roll=0.0):
    """Palm-down flange rotation: fingers (flange x) at world yaw, then roll about the finger axis."""
    return R.from_euler("z", yaw) * R.from_rotvec([np.pi, 0.0, 0.0]) * R.from_euler("x", roll)


def yaw_of(rot):
    x = rot.apply([1.0, 0.0, 0.0])
    return float(np.arctan2(x[1], x[0]))


def grasp_rotation(d, incline_rad):
    """Fingers horizontal and perpendicular to the handle, handle along flange +y toward its end
    (inclined down by incline_rad), palm normal (flange z) pointing down."""
    d = np.asarray(d, float)[:2]
    d = d / np.linalg.norm(d)
    x = np.array([-d[1], d[0], 0.0])
    c, s = np.cos(incline_rad), np.sin(incline_rad)
    y = np.array([c * d[0], c * d[1], -s])
    z = np.cross(x, y)
    if z[2] > 0:
        x = -x
        z = np.cross(x, y)
    return R.from_matrix(np.stack([x, y, z], 1))


def continuous_rotvec(rot, near):
    rv = rot.as_rotvec()
    if near is None:
        return rv
    ang = np.linalg.norm(rv)
    if ang < 1e-9:
        return rv
    axis = rv / ang
    best = rv
    for k in (-1, 1):
        alt = axis * (ang + 2 * np.pi * k)
        if np.linalg.norm(alt - near) < np.linalg.norm(best - near):
            best = alt
    return best


def row(p, rv, hand):
    return [float(v) for v in list(p) + list(rv) + list(hand)]


def interp_rows(p0, r0, h0, p1, r1, h1, n, near=None, ease=True):
    """n rows from (p0, r0, h0) to (p1, r1, h1); slerped rotation, continuous rotvecs."""
    n = max(1, int(n))
    sl = Slerp([0, 1], R.concatenate([r0, r1]))
    rows = []
    prev = near if near is not None else r0.as_rotvec()
    h0 = np.asarray(h0, float)
    h1 = np.asarray(h1, float)
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    for i in range(1, n + 1):
        s = i / n
        if ease:
            s = 0.5 - 0.5 * np.cos(np.pi * s)
        rv = continuous_rotvec(sl([s])[0], prev)
        prev = rv
        rows.append(row((1 - s) * p0 + s * p1, rv, (1 - s) * h0 + s * h1))
    return rows


def move_steps(p0, r0, p1, r1, v_max=0.25, w_max=1.0, n_min=10):
    """Rows needed for a smooth move at about v_max (m/s) and w_max (rad/s), 20 ms per row."""
    dist = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))
    ang = float((r1 * r0.inv()).magnitude())
    # cosine easing peaks at pi/2 times the mean speed
    t = max(dist / v_max, ang / w_max) * 1.57
    return max(n_min, int(np.ceil(t / 0.02)))
