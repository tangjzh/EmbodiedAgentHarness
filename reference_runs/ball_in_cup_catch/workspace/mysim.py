"""My own planar model of the public ball_in_cup geometry (pure Python, no simulator handle).

Built only from model.xml values + measurements in memory/exp*.npz. Used to screen controllers offline.
Integrator: semi-implicit Euler, h=0.002, 10 substeps per control tick, implicit joint damping (evidence:
first logged tick gives dz=-g*h^2*55 and vz=-g*0.02 for the free ball).
Constraints: soft acceleration-level model a1=(1-d)*a0+d*aref with timeconst .02, dampratio 1, d~.95.
"""
import math

H = 0.002
NSUB = 10
G = 9.81
R_BALL = 0.025
R_CAP = 0.008
ANCHOR_LOCAL = -0.108
STRING = 0.3
CAPS = [(-.05, 0.0, -.05, -.075), (-.05, -.075, -.025, -.1), (-.025, -.1, .025, -.1),
        (.025, -.1, .05, -.075), (.05, -.075, .05, 0.0)]
MB = 1000.0 * (4.0 / 3.0) * math.pi * R_BALL ** 3
_cap_len = sum(math.hypot(c[2] - c[0], c[3] - c[1]) for c in CAPS)
MC = 1000.0 * (math.pi * R_CAP ** 2 * _cap_len + 5 * (4.0 / 3.0) * math.pi * R_CAP ** 3)


class Params:
    def __init__(self, mc=MC, mb=MB, stiff=20.0, damp=3.0, gear=5.0, tc=0.02, zeta=1.0,
                 dmin=0.9, dmax=0.95, width=0.001, mu=1.0):
        self.mc, self.mb, self.stiff, self.damp, self.gear = mc, mb, stiff, damp, gear
        self.tc, self.zeta, self.dmin, self.dmax, self.width, self.mu = tc, zeta, dmin, dmax, width, mu


def _imp(r, P):
    """Impedance d(r) (approximate: linear blend over width, r = penetration depth >= 0)."""
    x = min(abs(r) / P.width, 1.0)
    return P.dmin + (P.dmax - P.dmin) * x


def _solve(a0, Jn, Ainv_diag, r, vn, P):
    """Scalar soft unilateral constraint: returns impulse-free force magnitude f>=0 along J.
    a0 = J*unconstrained accel, r = signed distance (<0 penetrating), vn = J*v."""
    d = _imp(-r, P)
    b = 2.0 / (P.dmax * P.tc)
    k = d / (P.dmax ** 2 * P.tc ** 2 * P.zeta ** 2)
    aref = -b * vn - k * r
    f = d * (aref - a0) * Ainv_diag
    return max(f, 0.0), d


class Sim:
    def __init__(self, ball_x=0.0, ball_z=0.3, P=None):
        self.P = P or Params()
        self.q = [0.0, 0.0, ball_x, ball_z]
        self.v = [0.0, 0.0, 0.0, 0.0]
        self.t = 0
        self.contact_hist = 0

    def state(self):
        return list(self.q), list(self.v)

    def in_target(self):
        cx, cz, bx, bz = self.q
        return abs(bx - cx) < 0.025 and abs((0.2 + bz) - (0.6 + cz - 0.05)) < 0.025

    def substep(self, ux, uz):
        P = self.P
        cx, cz, bx, bz = self.q
        vcx, vcz, vbx, vbz = self.v
        mc, mb = P.mc, P.mb
        imc, imb = 1.0 / mc, 1.0 / mb
        # smooth accelerations (damping explicit in force, implicit in the update below, as in Euler+eulerdamp)
        fcx = P.gear * ux - P.stiff * cx - P.damp * vcx
        fcz = P.gear * uz - P.stiff * cz - P.damp * vcz - mc * G
        acx, acz = fcx * imc, fcz * imc
        abx, abz = 0.0, -G
        A_pair = imc + imb
        Ainv_pair = 1.0 / A_pair
        wbz = 0.2 + bz
        wcz = 0.6 + cz
        # --- tendon limit ---
        ax = cx
        az = wcz + ANCHOR_LOCAL
        dx = bx - ax
        dz = wbz - az
        L = math.hypot(dx, dz)
        if L > STRING and L > 1e-9:
            nx, nz = dx / L, dz / L  # from anchor to ball
            # constraint coordinate: dist = STRING - L; J*v = -(v_ball - v_anchor).n
            rel_v = (vbx - vcx) * nx + (vbz - vcz) * nz
            rel_a = (abx - acx) * nx + (abz - acz) * nz
            f, _ = _solve(-rel_a, None, Ainv_pair, STRING - L, -rel_v, P)
            # f pulls ball toward anchor, anchor toward ball
            abx -= f * nx * imb
            abz -= f * nz * imb
            acx += f * nx * imc
            acz += f * nz * imc
        # --- ball vs cup capsules ---
        lx = bx - cx
        lz = wbz - wcz
        if -0.15 < lz < 0.05 and abs(lx) < 0.1:
            for (x1, z1, x2, z2) in CAPS:
                sx, sz = x2 - x1, z2 - z1
                ss = sx * sx + sz * sz
                tt = ((lx - x1) * sx + (lz - z1) * sz) / ss
                tt = 0.0 if tt < 0.0 else (1.0 if tt > 1.0 else tt)
                px, pz = x1 + tt * sx, z1 + tt * sz
                ex, ez = lx - px, lz - pz
                dist = math.hypot(ex, ez)
                r = dist - (R_BALL + R_CAP)
                if r < 0.0 and dist > 1e-9:
                    nx, nz = ex / dist, ez / dist  # from capsule to ball
                    rel_v = (vbx - vcx) * nx + (vbz - vcz) * nz
                    rel_a = (abx - acx) * nx + (abz - acz) * nz
                    f, d = _solve(rel_a, None, Ainv_pair, r, rel_v, P)
                    tx, tz = -nz, nx
                    rel_vt = (vbx - vcx) * tx + (vbz - vcz) * tz
                    rel_at = (abx - acx) * tx + (abz - acz) * tz
                    ft = -(rel_at + rel_vt / (2.0 * H)) * Ainv_pair * d
                    lim = P.mu * f
                    ft = -lim if ft < -lim else (lim if ft > lim else ft)
                    fbx = f * nx + ft * tx
                    fbz = f * nz + ft * tz
                    abx += fbx * imb
                    abz += fbz * imb
                    acx -= fbx * imc
                    acz -= fbz * imc
                    self.contact_hist += 1
        # --- ball vs ground plane (z=0) ---
        r = wbz - R_BALL
        if r < 0.0:
            f, d = _solve(abz, None, mb, r, vbz, P)
            lim = P.mu * f
            ft = -(abx + vbx / (2.0 * H)) * mb * d
            ft = -lim if ft < -lim else (lim if ft > lim else ft)
            abz += f * imb
            abx += ft * imb
        # implicit-damping update for cup joints; ball has no joint damping
        scale = mc / (mc + H * P.damp)
        vcx += H * acx * scale
        vcz += H * acz * scale
        vbx += H * abx
        vbz += H * abz
        self.v = [vcx, vcz, vbx, vbz]
        self.q = [cx + H * vcx, cz + H * vcz, bx + H * vbx, bz + H * vbz]

    def step(self, u):
        ux = max(-1.0, min(1.0, u[0]))
        uz = max(-1.0, min(1.0, u[1]))
        for _ in range(NSUB):
            self.substep(ux, uz)
        self.t += 1
        return self.in_target()
