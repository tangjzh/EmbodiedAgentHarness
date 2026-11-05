"""RGB-D perception for the Factory gear-mesh task (no object GT).

All three calibrated cameras are fused into a world point cloud. The target peg
(medium-gear shaft on the gear base) is found as the compact blob standing
~20 mm above the gear-base plate; its centre comes from its top cap.
"""
import numpy as np
from scipy import ndimage


def world_points(obs, cams, names=('scene', 'closeup', 'wrist')):
    pts, cols = [], []
    for n in names:
        cal = obs['cameras'][n]
        d = cams[n + '_depth_m']
        K = np.asarray(cal['intrinsic_cv'])
        T = np.asarray(cal['cam2world_cv'])
        off = cal.get('depth_pixel_ray_offset', [0.5, 0.5])
        H, W = d.shape
        u, v = np.meshgrid(np.arange(W) + off[0], np.arange(H) + off[1])
        m = np.isfinite(d) & (d > 0) & (d < 3)
        z = d[m]
        P = np.stack([(u[m] - K[0, 2]) / K[0, 0] * z, (v[m] - K[1, 2]) / K[1, 1] * z, z], -1)
        pts.append(P @ T[:3, :3].T + T[:3, 3])
        cols.append(cams[n + '_rgb'][m])
    return np.concatenate(pts), np.concatenate(cols)


def plate_height(P, center, radius=0.07):
    """Gear-base plate top: dominant height level between table and parts."""
    near = np.hypot(P[:, 0] - center[0], P[:, 1] - center[1]) < radius
    z = P[near, 2]
    z = z[(z > 0.008) & (z < 0.12)]
    h, e = np.histogram(z, bins=np.arange(0.008, 0.12, 0.001))
    k = np.argmax(h)
    zz = z[(z >= e[k] - 0.001) & (z < e[k + 1] + 0.001)]
    return float(np.median(zz))


def find_peg(P, guess_xy, plate_z, res=0.001, search=0.06):
    """Return dict with peg centre xy, top z and diagnostics."""
    dx = P[:, 0] - guess_xy[0]
    dy = P[:, 1] - guess_xy[1]
    rel = P[:, 2] - plate_z
    sel = (np.abs(dx) < search) & (np.abs(dy) < search) & (rel > 0.010) & (rel < 0.024)
    Q = P[sel]
    N = int(round(2 * search / res))
    i = np.clip(((Q[:, 0] - guess_xy[0] + search) / res).astype(int), 0, N - 1)
    j = np.clip(((Q[:, 1] - guess_xy[1] + search) / res).astype(int), 0, N - 1)
    occ = np.zeros((N, N), bool)
    occ[i, j] = True
    occ = ndimage.binary_closing(occ, iterations=1)
    lab, n = ndimage.label(occ)
    cands = []
    for k in range(1, n + 1):
        area = (lab == k).sum() * res * res * 1e6  # mm^2
        mk = lab[i, j] == k
        if mk.sum() < 10:
            continue
        q = Q[mk]
        top = q[:, 2].max() - plate_z
        ext = np.ptp(q[:, :2], axis=0).max()
        cands.append(dict(k=k, area=area, top=top, ext=ext, n=int(mk.sum()),
                          xy=q[:, :2].mean(0)))
    # medium peg: thin shaft whose top is ~20 mm above the plate (small-gear hub
    # is ~24+ mm, large gear ~25 mm) with nothing standing on it.
    pegs = []
    for c in cands:
        if not (15 < c['area'] < 200 and c['ext'] < 0.016 and 0.0175 < c['top'] < 0.0225):
            continue
        r = np.hypot(P[:, 0] - c['xy'][0], P[:, 1] - c['xy'][1])
        h = P[:, 2] - plate_z
        c['above'] = int(((r < 0.006) & (h > c['top'] + 0.002) & (h < 0.04)).sum())
        if c['above'] < 20:
            pegs.append(c)
    if not pegs:
        return dict(ok=False, cands=cands)
    best = min(pegs, key=lambda c: np.hypot(*(c['xy'] - guess_xy)))
    # Refine with a circle fit to the shaft's side surface (R ~ 4.7 mm). The top
    # cap centroid is biased ~1 mm by its rim; the side fit matched the final
    # inserted pose to 0.1 mm in development.
    xy = best['xy']
    h = P[:, 2] - plate_z
    for _ in range(3):
        r = np.hypot(P[:, 0] - xy[0], P[:, 1] - xy[1])
        side = (r < 0.008) & (h > 0.004) & (h < best['top'] - 0.003)
        if side.sum() < 30:
            break
        c, R = fit_circle(P[side, 0], P[side, 1])
        if not 0.0035 < R < 0.006:
            break
        xy = c
    best.update(ok=True, xy=xy, side_n=int(side.sum()), top_z=plate_z + best['top'])
    return best


def fit_circle(x, y):
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    s = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    return s[:2], float(np.sqrt(s[2] + s[0] ** 2 + s[1] ** 2))


def held_gear(P, C, eef):
    """Held gear relative to EEF: axis xy offset (circle fit to the blue tooth
    ring, R ~ 20.2 mm) and depth of its lowest face below the EEF."""
    C = C.astype(int)
    blue = (C[:, 2] - C[:, 0] > 40) & (np.hypot(P[:, 0] - eef[0], P[:, 1] - eef[1]) < 0.03)
    blue &= np.abs(P[:, 2] - eef[2]) < 0.03
    out = dict(offset=None, bottom=None)
    if blue.sum() < 50:
        return out
    out['bottom'] = float(eef[2] - np.percentile(P[blue, 2], 1) + 0.0003)
    lo = eef[2] - out['bottom']
    band = blue & (P[:, 2] > lo + 0.002) & (P[:, 2] < lo + 0.007)
    if band.sum() < 50:
        return out
    c, R = fit_circle(P[band, 0], P[band, 1])
    if 0.018 < R < 0.022:
        out['offset'] = (c - np.asarray(eef[:2])).tolist()
    out['R'] = R
    return out
