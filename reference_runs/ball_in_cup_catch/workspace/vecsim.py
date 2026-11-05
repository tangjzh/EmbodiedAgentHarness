"""Batch (NumPy) version of mysim.py: N independent copies of my planar model stepped in parallel."""
import numpy as np
import mysim

H, NSUB, G = mysim.H, mysim.NSUB, mysim.G
R_BALL, R_CAP, ANCHOR, STRING = mysim.R_BALL, mysim.R_CAP, mysim.ANCHOR_LOCAL, mysim.STRING
CAPS = np.array(mysim.CAPS)
MB, MC = mysim.MB, mysim.MC


class VP:
    stiff, damp, gear = 20.0, 3.0, 5.0
    tc, zeta, dmin, dmax, width, mu = 0.02, 1.0, 0.9, 0.95, 0.001, 1.0
    mc, mb = MC, MB


def _soft(a0, Ainv, r, vn, P):
    d = P.dmin + (P.dmax - P.dmin) * np.minimum(np.abs(r) / P.width, 1.0)
    b = 2.0 / (P.dmax * P.tc)
    k = d / (P.dmax ** 2 * P.tc ** 2 * P.zeta ** 2)
    aref = -b * vn - k * r
    return np.maximum(d * (aref - a0) * Ainv, 0.0), d


def substep(q, v, u, P=VP):
    cx, cz, bx, bz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vcx, vcz, vbx, vbz = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
    mc, mb = P.mc, P.mb
    imc, imb = 1.0 / mc, 1.0 / mb
    Ainv = 1.0 / (imc + imb)
    acx = (P.gear * u[:, 0] - P.stiff * cx - P.damp * vcx) * imc
    acz = (P.gear * u[:, 1] - P.stiff * cz - P.damp * vcz) * imc - G
    abx = np.zeros_like(cx)
    abz = np.full_like(cx, -G)
    wbz = 0.2 + bz
    wcz = 0.6 + cz
    # tendon
    dx = bx - cx
    dz = wbz - (wcz + ANCHOR)
    Lr = np.sqrt(dx * dx + dz * dz) + 1e-12
    taut = Lr > STRING
    if taut.any():
        nx, nz = dx / Lr, dz / Lr
        rel_v = (vbx - vcx) * nx + (vbz - vcz) * nz
        rel_a = (abx - acx) * nx + (abz - acz) * nz
        f, _ = _soft(-rel_a, Ainv, STRING - Lr, -rel_v, P)
        f = np.where(taut, f, 0.0)
        abx = abx - f * nx * imb
        abz = abz - f * nz * imb
        acx = acx + f * nx * imc
        acz = acz + f * nz * imc
    # ball vs cup capsules
    lx = bx - cx
    lz = wbz - wcz
    near = (lz > -0.15) & (lz < 0.05) & (np.abs(lx) < 0.1)
    if near.any():
        for (x1, z1, x2, z2) in CAPS:
            sx, sz = x2 - x1, z2 - z1
            ss = sx * sx + sz * sz
            tt = np.clip(((lx - x1) * sx + (lz - z1) * sz) / ss, 0.0, 1.0)
            ex = lx - (x1 + tt * sx)
            ez = lz - (z1 + tt * sz)
            dist = np.sqrt(ex * ex + ez * ez) + 1e-12
            r = dist - (R_BALL + R_CAP)
            hit = near & (r < 0.0)
            if not hit.any():
                continue
            nx, nz = ex / dist, ez / dist
            rel_v = (vbx - vcx) * nx + (vbz - vcz) * nz
            rel_a = (abx - acx) * nx + (abz - acz) * nz
            f, d = _soft(rel_a, Ainv, r, rel_v, P)
            tx, tz = -nz, nx
            rel_vt = (vbx - vcx) * tx + (vbz - vcz) * tz
            rel_at = (abx - acx) * tx + (abz - acz) * tz
            ft = np.clip(-(rel_at + rel_vt / (2.0 * H)) * Ainv * d, -P.mu * f, P.mu * f)
            f = np.where(hit, f, 0.0)
            ft = np.where(hit, ft, 0.0)
            fbx = f * nx + ft * tx
            fbz = f * nz + ft * tz
            abx = abx + fbx * imb
            abz = abz + fbz * imb
            acx = acx - fbx * imc
            acz = acz - fbz * imc
    # ground
    r = wbz - R_BALL
    gh = r < 0.0
    if gh.any():
        f, d = _soft(abz, mb, r, vbz, P)
        ft = np.clip(-(abx + vbx / (2.0 * H)) * mb * d, -P.mu * f, P.mu * f)
        abz = abz + np.where(gh, f, 0.0) * imb
        abx = abx + np.where(gh, ft, 0.0) * imb
    scale = mc / (mc + H * P.damp)
    vn = np.stack([vcx + H * acx * scale, vcz + H * acz * scale, vbx + H * abx, vbz + H * abz], axis=1)
    qn = q + H * vn
    return qn, vn


def step(q, v, u, P=VP):
    u = np.clip(u, -1.0, 1.0)
    for _ in range(NSUB):
        q, v = substep(q, v, u, P)
    return q, v


def in_target(q):
    return (np.abs(q[:, 2] - q[:, 0]) < 0.025) & (np.abs((0.2 + q[:, 3]) - (0.55 + q[:, 1])) < 0.025)


def rollout(q0, v0, U, P=VP):
    """q0,v0: (N,4); U: (N,T,2). Returns Q (N,T,4) after each tick, V likewise."""
    N, T, _ = U.shape
    Q = np.empty((N, T, 4))
    V = np.empty((N, T, 4))
    q, v = q0.copy(), v0.copy()
    for t in range(T):
        q, v = step(q, v, U[:, t], P)
        Q[:, t], V[:, t] = q, v
    return Q, V
