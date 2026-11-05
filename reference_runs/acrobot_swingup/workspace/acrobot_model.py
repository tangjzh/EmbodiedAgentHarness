"""Analytic acrobot model built only from public parameters in model.xml.

Angles: q1 = shoulder, q2 = elbow, q=0 upright, positive rotation about +y.
State x = [q1, q2, dq1, dq2]. Vectorised over a leading batch dimension.
"""
import numpy as np

G = 9.81
GEAR = 2.0
DAMP = 0.05
DT = 0.01
L1 = 1.0
LC1 = 0.5
LC2 = 0.5
M1 = 1.0
M2 = 1.0


def capsule_inertia(mass, radius, length):
    """Transverse inertia about the COM of a capsule of the given total mass."""
    v_sphere = 4.0 * np.pi * radius ** 3 / 3.0
    v_cyl = np.pi * radius ** 2 * length
    m_sphere = mass * v_sphere / (v_sphere + v_cyl)
    m_cyl = mass * v_cyl / (v_sphere + v_cyl)
    inertia = m_cyl * (3 * radius ** 2 + length ** 2) / 12.0
    inertia += 2 * m_sphere * radius ** 2 / 5.0
    inertia += m_sphere * length * (3 * radius + 2 * length) / 8.0
    return inertia


I1 = capsule_inertia(M1, 0.05, 1.0)
I2 = capsule_inertia(M2, 0.049, 1.0)


def deriv(x, u, params=None):
    """Continuous dynamics. x[..., 4], u[...] normalised action."""
    p = params or {}
    i1 = p.get("I1", I1)
    i2 = p.get("I2", I2)
    damp = p.get("damp", DAMP)
    gear = p.get("gear", GEAR)
    q1, q2, d1, d2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    c2 = np.cos(q2)
    s2 = np.sin(q2)
    s1 = np.sin(q1)
    s12 = np.sin(q1 + q2)
    a = M2 * L1 * LC2
    m11 = i1 + i2 + M1 * LC1 ** 2 + M2 * (L1 ** 2 + LC2 ** 2) + 2 * a * c2
    m12 = i2 + M2 * LC2 ** 2 + a * c2
    m22 = i2 + M2 * LC2 ** 2
    # generalised forces: gravity + coriolis + damping + actuator
    f1 = G * ((M1 * LC1 + M2 * L1) * s1 + M2 * LC2 * s12) + a * s2 * (2 * d1 * d2 + d2 ** 2) - damp * d1
    f2 = G * M2 * LC2 * s12 - a * s2 * d1 ** 2 - damp * d2 + gear * u
    det = m11 * m22 - m12 ** 2
    dd1 = (m22 * f1 - m12 * f2) / det
    dd2 = (-m12 * f1 + m11 * f2) / det
    return np.stack([d1, d2, dd1, dd2], axis=-1)


def step(x, u, dt=DT, params=None):
    """One RK4 step with zero-order-hold action."""
    k1 = deriv(x, u, params)
    k2 = deriv(x + 0.5 * dt * k1, u, params)
    k3 = deriv(x + 0.5 * dt * k2, u, params)
    k4 = deriv(x + dt * k3, u, params)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def obs_to_state(obs):
    o = np.asarray(obs["orientations"], dtype=float)
    v = np.asarray(obs["velocity"], dtype=float)
    q1 = np.arctan2(o[0], o[2])
    q12 = np.arctan2(o[1], o[3])
    q2 = np.arctan2(np.sin(q12 - q1), np.cos(q12 - q1))
    return np.array([q1, q2, v[0], v[1]])


def tip_distance(x):
    q1, q2 = x[..., 0], x[..., 1]
    tx = np.sin(q1) + np.sin(q1 + q2)
    tz = 2.0 + np.cos(q1) + np.cos(q1 + q2)
    return np.sqrt(tx ** 2 + (tz - 4.0) ** 2)


def reward(x):
    """dm_control tolerance with gaussian sigmoid, value 0.1 at the margin."""
    d = tip_distance(x)
    scale = np.sqrt(-2.0 * np.log(0.1))
    out = np.exp(-0.5 * ((d - 0.2) / 1.0 * scale) ** 2)
    return np.where(d <= 0.2, 1.0, out)


def energy(x):
    q1, q2, d1, d2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    a = M2 * L1 * LC2
    c2 = np.cos(q2)
    m11 = I1 + I2 + M1 * LC1 ** 2 + M2 * (L1 ** 2 + LC2 ** 2) + 2 * a * c2
    m12 = I2 + M2 * LC2 ** 2 + a * c2
    m22 = I2 + M2 * LC2 ** 2
    kin = 0.5 * (m11 * d1 ** 2 + 2 * m12 * d1 * d2 + m22 * d2 ** 2)
    pot = G * ((M1 * LC1 + M2 * L1) * np.cos(q1) + M2 * LC2 * np.cos(q1 + q2))
    return kin + pot


E_UP = G * (M1 * LC1 + M2 * L1 + M2 * LC2)
