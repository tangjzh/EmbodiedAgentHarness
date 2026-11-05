"""Franka Panda kinematics (standard published modified-DH parameters) for reachability checks.
Robot base at (-0.615, 0, 0), axes aligned with world (PUBLIC_MODEL.txt)."""
import numpy as np

BASE = np.array([-0.615, 0.0, 0.0])
# (a_{i-1}, d_i, alpha_{i-1})
DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
      (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]
Q_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def _mdh(a, d, al, th):
    ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1.0]])


def fk(q, upto=None):
    T = np.eye(4)
    T[:3, 3] = BASE
    frames = []
    for i, (a, d, al) in enumerate(DH):
        T = T @ _mdh(a, d, al, q[i])
        frames.append(T.copy())
    T = T @ _mdh(0, 0.107, 0, 0)             # flange
    T = T @ _mdh(0, 0, 0, -np.pi / 4)         # hand
    T = T @ _mdh(0, 0.1034, 0, 0)             # tcp
    return T, frames


def jac(q, eps=1e-6):
    """Geometric Jacobian (world frame, linear at TCP; angular)."""
    T, fr = fk(q)
    p = T[:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        z = fr[i][:3, 2]
        o = fr[i][:3, 3]
        J[:3, i] = np.cross(z, p - o)
        J[3:, i] = z
    return J


def ik(p_t, R_t, q0, iters=200, tol=1e-4, lam=0.05):
    """Damped least squares IK with joint-limit clamping. Returns (q, ok, err)."""
    from geom import rotvec
    q = np.array(q0, float)
    for _ in range(iters):
        T, _ = fk(q)
        e = np.r_[p_t - T[:3, 3], rotvec(R_t @ T[:3, :3].T)]
        if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < 10 * tol:
            return q, True, e
        J = jac(q)
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > 0.2:
            dq *= 0.2 / n
        q = np.clip(q + dq, Q_LO, Q_HI)
    return q, False, e
