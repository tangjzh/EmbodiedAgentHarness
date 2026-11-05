"""Ball-in-cup catch controller v2: gentle hold when the ball is in the cup, otherwise receding-horizon CEM over
my own planar model of the public model.xml. Self-contained: uses only the observation argument and these constants."""
import math
import numpy as np

G = 9.81
R_BALL, R_CAP = 0.025, 0.008
ANCHOR, STRING = -0.108, 0.3
CAPS = ((-.05, 0.0, -.05, -.075), (-.05, -.075, -.025, -.1), (-.025, -.1, .025, -.1),
        (.025, -.1, .05, -.075), (.05, -.075, .05, 0.0))
MB = 1000.0 * (4.0 / 3.0) * math.pi * R_BALL ** 3
MC = 1000.0 * (math.pi * R_CAP ** 2 * sum(math.hypot(c[2] - c[0], c[3] - c[1]) for c in CAPS)
               + 5 * (4.0 / 3.0) * math.pi * R_CAP ** 3)
STIFF, DAMP, GEAR = 20.0, 3.0, 5.0
TC, DMIN, DMAX, WIDTH, MU = 0.02, 0.9, 0.95, 0.001, 1.0
HANG_Z = -(MC + MB) * G / STIFF

PAR = {"T": 55, "knot": 3, "N": 64, "elite": 8, "iters0": 10, "iters": 2, "std0": 0.8, "std": 0.5,
       "sub": 5, "seed": 0, "w_in": 10.0, "w_dist": 20.0, "w_lz": 5.0, "w_end": 20.0, "alpha": 0.7}


def _sat(x):
    return -1.0 if x < -1.0 else (1.0 if x > 1.0 else x)


def _in_cup(s):
    lx, lz = s[2] - s[0], (0.2 + s[3]) - (0.6 + s[1])
    return abs(lx) < 0.05 and -0.1 < lz < 0.0


def _hold(cx, cz):
    """Gentle hold: spring equilibrium is at most 1 cm from the current pose, toward the rest pose."""
    x_eq = cx + float(np.clip(-cx, -0.01, 0.01))
    z_eq = cz + float(np.clip(HANG_Z - cz, -0.01, 0.01))
    return _sat(STIFF * x_eq / GEAR), _sat((STIFF * z_eq + (MC + MB) * G) / GEAR)


def _soft(a0, r, vn, h_ainv):
    d = DMIN + (DMAX - DMIN) * np.minimum(np.abs(r) / WIDTH, 1.0)
    aref = -(2.0 / (DMAX * TC)) * vn - d / (DMAX ** 2 * TC ** 2) * r
    return np.maximum(d * (aref - a0) * h_ainv, 0.0), d


def rollout(s, U, par):
    """Batch rollout from state tuple s under plans U (N,T,2). Returns score (N,) and in-target tick counts."""
    N, T = U.shape[0], U.shape[1]
    cx, cz, bx, bz, vcx, vcz, vbx, vbz = (np.full(N, float(x)) for x in s)
    sub = par["sub"]
    h = 0.02 / sub
    imc, imb = 1.0 / MC, 1.0 / MB
    ainv = 1.0 / (imc + imb)
    scale = MC / (MC + h * DAMP)
    cnt = np.zeros(N)
    mind = np.full(N, 9.0)
    maxlz = np.full(N, -9.0)
    inside = np.zeros(N, bool)
    for t in range(T):
        lx = bx - cx
        lz = 0.2 + bz - 0.6 - cz
        cup = (np.abs(lx) < 0.05) & (lz > -0.1) & (lz < 0.0)
        # Plans are overridden by the hold rule inside the cup, exactly as the real policy behaves.
        ux = np.where(cup, np.clip(STIFF * (cx + np.clip(-cx, -.01, .01)) / GEAR, -1, 1), U[:, t, 0])
        uz = np.where(cup, np.clip((STIFF * (cz + np.clip(HANG_Z - cz, -.01, .01)) + (MC + MB) * G) / GEAR, -1, 1),
                      U[:, t, 1])
        for _ in range(sub):
            acx = (GEAR * ux - STIFF * cx - DAMP * vcx) * imc
            acz = (GEAR * uz - STIFF * cz - DAMP * vcz) * imc - G
            abx = np.zeros(N)
            abz = np.full(N, -G)
            wbz = 0.2 + bz
            wcz = 0.6 + cz
            dx = bx - cx
            dz = wbz - wcz - ANCHOR
            L = np.sqrt(dx * dx + dz * dz) + 1e-12
            taut = L > STRING
            if taut.any():
                nx, nz = dx / L, dz / L
                vn = -((vbx - vcx) * nx + (vbz - vcz) * nz)
                a0 = -((abx - acx) * nx + (abz - acz) * nz)
                f, _ = _soft(a0, STRING - L, vn, ainv)
                f = np.where(taut, f, 0.0)
                abx = abx - f * nx * imb
                abz = abz - f * nz * imb
                acx = acx + f * nx * imc
                acz = acz + f * nz * imc
            lx = bx - cx
            lz = wbz - wcz
            near = (lz > -0.15) & (lz < 0.05) & (np.abs(lx) < 0.1)
            if near.any():
                for (x1, z1, x2, z2) in CAPS:
                    sx, sz = x2 - x1, z2 - z1
                    tt = np.clip(((lx - x1) * sx + (lz - z1) * sz) / (sx * sx + sz * sz), 0.0, 1.0)
                    ex = lx - (x1 + tt * sx)
                    ez = lz - (z1 + tt * sz)
                    dist = np.sqrt(ex * ex + ez * ez) + 1e-12
                    r = dist - (R_BALL + R_CAP)
                    hit = near & (r < 0.0)
                    if not hit.any():
                        continue
                    nx, nz = ex / dist, ez / dist
                    vn = (vbx - vcx) * nx + (vbz - vcz) * nz
                    a0 = (abx - acx) * nx + (abz - acz) * nz
                    f, d = _soft(a0, r, vn, ainv)
                    tx, tz = -nz, nx
                    vt = (vbx - vcx) * tx + (vbz - vcz) * tz
                    at = (abx - acx) * tx + (abz - acz) * tz
                    ft = np.clip(-(at + vt / (2.0 * h)) * ainv * d, -MU * f, MU * f)
                    f = np.where(hit, f, 0.0)
                    ft = np.where(hit, ft, 0.0)
                    fbx = f * nx + ft * tx
                    fbz = f * nz + ft * tz
                    abx = abx + fbx * imb
                    abz = abz + fbz * imb
                    acx = acx - fbx * imc
                    acz = acz - fbz * imc
            r = wbz - R_BALL
            gh = r < 0.0
            if gh.any():
                f, d = _soft(abz, r, vbz, MB)
                ft = np.clip(-(abx + vbx / (2.0 * h)) * MB * d, -MU * f, MU * f)
                abz = abz + np.where(gh, f, 0.0) * imb
                abx = abx + np.where(gh, ft, 0.0) * imb
            vcx = vcx + h * acx * scale
            vcz = vcz + h * acz * scale
            vbx = vbx + h * abx
            vbz = vbz + h * abz
            cx = cx + h * vcx
            cz = cz + h * vcz
            bx = bx + h * vbx
            bz = bz + h * vbz
        lx = bx - cx
        lz = 0.2 + bz - 0.6 - cz
        inside = (np.abs(lx) < 0.025) & (np.abs(lz + 0.05) < 0.025)
        cnt += inside
        mind = np.minimum(mind, np.abs(lx) + np.abs(lz + 0.05))
        maxlz = np.maximum(maxlz, lz)
    score = (par["w_in"] * cnt - par["w_dist"] * mind + par["w_lz"] * np.minimum(maxlz, 0.1)
             + par["w_end"] * inside)
    return score, cnt


def _interp(par):
    T, K = par["T"], par["knot"]
    n = T // K + 2
    idx = np.arange(T) / float(K)
    lo = idx.astype(int)
    return lo, lo + 1, (idx - lo)[None, :, None], n


def plan(s, memory, par):
    rng = memory.setdefault("rng", np.random.default_rng(par["seed"]))
    lo, hi, w, n = _interp(par)
    T, N = par["T"], par["N"]
    fresh = "mean" not in memory or memory.get("stale", True)
    if fresh:
        mean = np.zeros((T, 2))
        best = None
        iters, std = par["iters0"], par["std0"]
    else:
        mean = np.vstack([memory["mean"][1:], memory["mean"][-1:] * 0.0])
        best = np.vstack([memory["best"][1:], memory["best"][-1:] * 0.0])
        iters, std = par["iters"], par["std"]
    best_score = -np.inf
    best_cnt = 0.0
    for it in range(iters):
        nk = rng.standard_normal((N, n, 2)) * std
        U = np.clip(mean + nk[:, lo] * (1 - w) + nk[:, hi] * w, -1.0, 1.0)
        U[0] = mean
        if best is not None:
            U[1] = best
        score, cnt = rollout(s, U, par)
        order = np.argsort(-score)
        if score[order[0]] > best_score:
            best_score, best, best_cnt = float(score[order[0]]), U[order[0]].copy(), float(cnt[order[0]])
        mean = (1 - par["alpha"]) * mean + par["alpha"] * U[order[:par["elite"]]].mean(0)
    memory["mean"], memory["best"], memory["stale"] = mean, best, False
    memory["pred_in"] = best_cnt
    return best[0]


def policy(observation, step, memory):
    obs = observation.get("observation", observation) if isinstance(observation, dict) else observation
    q, v = obs["position"], obs["velocity"]
    s = tuple(float(x) for x in list(q) + list(v))
    par = memory.get("par", PAR)
    if _in_cup(s):
        memory["mode"] = "hold"
        memory["stale"] = True
        return list(_hold(s[0], s[1]))
    memory["mode"] = "mpc"
    a = plan(s, memory, par)
    return [float(a[0]), float(a[1])]
