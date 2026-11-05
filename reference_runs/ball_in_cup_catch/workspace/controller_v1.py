"""Ball-in-cup catch controller. Self-contained: uses only the observation and constants from public model.xml.

Layers: (1) hold when the ball is in the cup, (2) a catch primitive (cup drops, tracks predicted ball x, keeps wall
clearance until the ball is over the rim), used only when an internal forward rollout predicts success,
(3) otherwise a phase-based swing pump in x plus tension-synchronised pumping in z.
"""
import math

H = 0.002
NSUB = 10
G = 9.81
R_BALL = 0.025
R_CAP = 0.008
ANCHOR = -0.108
STRING = 0.3
CAPS = ((-.05, 0.0, -.05, -.075), (-.05, -.075, -.025, -.1), (-.025, -.1, .025, -.1),
        (.025, -.1, .05, -.075), (.05, -.075, .05, 0.0))
MB = 1000.0 * (4.0 / 3.0) * math.pi * R_BALL ** 3
MC = 1000.0 * (math.pi * R_CAP ** 2 * sum(math.hypot(c[2] - c[0], c[3] - c[1]) for c in CAPS)
               + 5 * (4.0 / 3.0) * math.pi * R_CAP ** 3)
STIFF, DAMP, GEAR = 20.0, 3.0, 5.0
TC, DMIN, DMAX, WIDTH, MU = 0.02, 0.9, 0.95, 0.001, 1.0
HANG_Z = -(MC + MB) * G / STIFF

PAR = {
    "kx": 50.0, "lam": 5.5,            # x pump: u = sat(kx*(v_rel - lam*r_rel))
    "kz": 0.25, "T0": 14.0,            # z pump: u = sat(kz*(tension_estimate - T0))
    "catch_kp": 400.0, "catch_kd": 40.0,
    "lead": 0.04,                      # s of ball-x prediction in the catch primitive
    "clear": 0.09,                     # lateral clearance from the wall while the ball is below the rim
    "over": 0.036,                     # ball centre height above rim plane counted as "over the rim"
    "horizon": 45,                     # verification rollout length, ticks
    "use_rollout": True,
}


def _sat(x):
    return -1.0 if x < -1.0 else (1.0 if x > 1.0 else x)


def _tick(s, ux, uz):
    """One control tick of my planar model. s = (cx,cz,bx,bz,vcx,vcz,vbx,vbz)."""
    cx, cz, bx, bz, vcx, vcz, vbx, vbz = s
    imc, imb = 1.0 / MC, 1.0 / MB
    ainv = 1.0 / (imc + imb)
    scale = MC / (MC + H * DAMP)
    b = 2.0 / (DMAX * TC)
    ku = 1.0 / (DMAX ** 2 * TC ** 2)
    for _ in range(NSUB):
        acx = (GEAR * ux - STIFF * cx - DAMP * vcx) * imc
        acz = (GEAR * uz - STIFF * cz - DAMP * vcz) * imc - G
        abx, abz = 0.0, -G
        wbz = 0.2 + bz
        wcz = 0.6 + cz
        dx = bx - cx
        dz = wbz - (wcz + ANCHOR)
        L = math.sqrt(dx * dx + dz * dz)
        if L > STRING:
            nx, nz = dx / L, dz / L
            r = STRING - L
            d = DMIN + (DMAX - DMIN) * min(-r / WIDTH, 1.0)
            a0 = -((abx - acx) * nx + (abz - acz) * nz)
            vn = -((vbx - vcx) * nx + (vbz - vcz) * nz)
            f = d * (-b * vn - d * ku * r - a0) * ainv
            if f > 0.0:
                abx -= f * nx * imb
                abz -= f * nz * imb
                acx += f * nx * imc
                acz += f * nz * imc
        lx = bx - cx
        lz = wbz - wcz
        if -0.15 < lz < 0.05 and -0.1 < lx < 0.1:
            for (x1, z1, x2, z2) in CAPS:
                sx, sz = x2 - x1, z2 - z1
                tt = ((lx - x1) * sx + (lz - z1) * sz) / (sx * sx + sz * sz)
                tt = 0.0 if tt < 0.0 else (1.0 if tt > 1.0 else tt)
                ex, ez = lx - (x1 + tt * sx), lz - (z1 + tt * sz)
                dist = math.sqrt(ex * ex + ez * ez)
                r = dist - (R_BALL + R_CAP)
                if r < 0.0 and dist > 1e-9:
                    nx, nz = ex / dist, ez / dist
                    d = DMIN + (DMAX - DMIN) * min(-r / WIDTH, 1.0)
                    vn = (vbx - vcx) * nx + (vbz - vcz) * nz
                    a0 = (abx - acx) * nx + (abz - acz) * nz
                    f = d * (-b * vn - d * ku * r - a0) * ainv
                    if f > 0.0:
                        tx, tz = -nz, nx
                        vt = (vbx - vcx) * tx + (vbz - vcz) * tz
                        at = (abx - acx) * tx + (abz - acz) * tz
                        ft = -(at + vt / (2.0 * H)) * ainv * d
                        lim = MU * f
                        ft = -lim if ft < -lim else (lim if ft > lim else ft)
                        fbx, fbz = f * nx + ft * tx, f * nz + ft * tz
                        abx += fbx * imb
                        abz += fbz * imb
                        acx -= fbx * imc
                        acz -= fbz * imc
        r = wbz - R_BALL
        if r < 0.0:
            d = DMIN + (DMAX - DMIN) * min(-r / WIDTH, 1.0)
            f = d * (-b * vbz - d * ku * r - abz) * MB
            if f > 0.0:
                ft = -(abx + vbx / (2.0 * H)) * MB * d
                lim = MU * f
                ft = -lim if ft < -lim else (lim if ft > lim else ft)
                abz += f * imb
                abx += ft * imb
        vcx += H * acx * scale
        vcz += H * acz * scale
        vbx += H * abx
        vbz += H * abz
        cx += H * vcx
        cz += H * vcz
        bx += H * vbx
        bz += H * vbz
    return (cx, cz, bx, bz, vcx, vcz, vbx, vbz)


def _local(s):
    return s[2] - s[0], (0.2 + s[3]) - (0.6 + s[1])


def _in_cup(s):
    lx, lz = _local(s)
    return abs(lx) < 0.05 and -0.1 < lz < 0.0


def _in_target(s):
    lx, lz = _local(s)
    return abs(lx) < 0.025 and abs(lz + 0.05) < 0.025


def _hold(s):
    """Memoryless gentle hold: spring equilibrium sits at most 1 cm from the current pose, toward the rest pose."""
    cx, cz = s[0], s[1]
    ex = -cx
    ez = HANG_Z - cz
    x_eq = cx + (-0.01 if ex < -0.01 else (0.01 if ex > 0.01 else ex))
    z_eq = cz + (-0.01 if ez < -0.01 else (0.01 if ez > 0.01 else ez))
    return _sat(STIFF * x_eq / GEAR), _sat((STIFF * z_eq + (MC + MB) * G) / GEAR)


def _catch(s, par, zmode="drop"):
    """Catch primitive: pure function of state."""
    cx, cz, bx, bz, vcx, vcz, vbx, vbz = s
    lx, lz = _local(s)
    over = lz - par["over"]
    xd = bx + vbx * par["lead"]
    vd = vbx
    if over < 0.0:
        # Ball below the rim plane: approach, but keep the cup wall clear of the ball.
        c = par["clear"]
        side = 1.0 if lx >= 0.0 else -1.0
        xd = bx - side * c
        vd = 0.0
    ax = par["catch_kp"] * (xd - cx) + par["catch_kd"] * (vd - vcx)
    ux = _sat((MC * ax + DAMP * vcx + STIFF * cx) / GEAR)
    if zmode == "drop":
        uz = -1.0
    else:  # hold the current height against gravity
        uz = _sat((STIFF * cz + DAMP * vcz + MC * G) / GEAR)
    return ux, uz


def _pump(s, par):
    cx, cz, bx, bz, vcx, vcz, vbx, vbz = s
    rx = bx - cx
    rz = (0.2 + bz) - (0.6 + cz + ANCHOR)
    L = math.sqrt(rx * rx + rz * rz) + 1e-9
    vrx = vbx - vcx
    vrz = vbz - vcz
    ux = _sat(par["kx"] * (vrx - par["lam"] * rx))
    tens = (vrx * vrx + vrz * vrz) / L - G * rz / L if L > 0.29 else 0.0
    uz = _sat(par["kz"] * (tens - par["T0"]))
    return ux, uz


def _rollout_ok(s, par, zmode):
    """Run the catch primitive through my model; True if the ball ends up settled inside the target."""
    n_in = 0
    for k in range(par["horizon"]):
        if _in_cup(s):
            ux, uz = _hold(s)
        else:
            ux, uz = _catch(s, par, zmode)
        s = _tick(s, ux, uz)
        n_in = n_in + 1 if _in_target(s) else 0
        if n_in >= 8:
            return True
    return False


def policy(observation, step, memory):
    obs = observation.get("observation", observation) if isinstance(observation, dict) else observation
    q, v = obs["position"], obs["velocity"]
    s = (float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(v[0]), float(v[1]), float(v[2]), float(v[3]))
    par = memory.get("par", PAR)
    if _in_cup(s):
        memory["mode"] = "hold"
        return list(_hold(s))
    if par["use_rollout"]:
        for zmode in ("drop", "keep"):
            if _rollout_ok(s, par, zmode):
                memory["mode"] = "catch_" + zmode
                return list(_catch(s, par, zmode))
    memory["mode"] = "pump"
    return list(_pump(s, par))
