"""Motion helpers shared by experiments (development copy; controller has its own)."""
import json

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

HAND_OPEN = [0.0] * 12 + [0.263, 0.0, 0.0, 0.0]


def state_pose(state):
    p = np.array(state[0:3])
    q = state[3:7]
    rot = R.from_quat([q[1], q[2], q[3], q[0]])
    return p, rot, list(state[7:23])


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


def interp_rows(p0, r0, h0, p1, r1, h1, n, near=None, ease=True):
    """n rows from (p0,r0,h0) to (p1,r1,h1); rotations slerped, rotvecs kept continuous."""
    sl = Slerp([0, 1], R.concatenate([r0, r1]))
    rows = []
    prev = near if near is not None else r0.as_rotvec()
    h0 = np.asarray(h0, float)
    h1 = np.asarray(h1, float)
    for i in range(1, n + 1):
        s = i / n
        if ease:
            s = 0.5 - 0.5 * np.cos(np.pi * s)
        p = (1 - s) * np.asarray(p0) + s * np.asarray(p1)
        rv = continuous_rotvec(sl([s])[0], prev)
        prev = rv
        h = (1 - s) * h0 + s * h1
        rows.append([float(v) for v in list(p) + list(rv) + list(h)])
    return rows


def meta_of(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except ValueError:
                pass
    return {}
