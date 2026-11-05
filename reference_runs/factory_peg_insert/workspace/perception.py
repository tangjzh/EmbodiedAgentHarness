"""RGB-D perception for Factory peg insertion: socket centre and held-peg pose from calibrated cameras."""
import numpy as np

CAMS = ("scene", "closeup", "wrist")


def backproject(depth, K, T):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    P = np.stack([x, y, depth, np.ones_like(depth)], -1) @ np.asarray(T).T
    return P[..., :3]


def yellow_mask(rgb):
    rgb = rgb.astype(np.int32)
    return (rgb[..., 0] > 120) & (rgb[..., 1] > 110) & (rgb[..., 2] < 90) & (rgb[..., 0] - rgb[..., 2] > 80)


def gather(cams, calib, cams_used=CAMS):
    """Return dict cam -> (points HxWx3, yellow mask, camera centre)."""
    out = {}
    for c in cams_used:
        K = np.asarray(calib[c]["intrinsic_cv"]); T = np.asarray(calib[c]["cam2world_cv"])
        D = cams[f"{c}_depth_m"]
        P = backproject(D, K, T)
        ok = np.isfinite(D) & (D > 0.02) & (D < 3.0)
        out[c] = (P, yellow_mask(cams[f"{c}_rgb"]) & ok, ok, T[:3, 3])
    return out


def table_z(data):
    zs = np.concatenate([P[ok][:, 2] for P, _, ok, _ in data.values()])
    zs = zs[(zs > -0.2) & (zs < 0.3)]
    h, e = np.histogram(zs, bins=500, range=(-0.2, 0.3))
    i = np.argmax(h)
    return 0.5 * (e[i] + e[i + 1])


def peg_points(data, tz):
    pts = [P[y][P[y][:, 2] > tz + 0.01] for P, y, _, _ in data.values()]
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def estimate(cams, calib, eef_xy, radius=0.04):
    """Estimate socket top centre and peg tip. Returns dict (world frame, metres)."""
    data = gather(cams, calib)
    tz = table_z(data)
    peg = peg_points(data, tz)
    res = {"table_z": float(tz), "n_peg": int(len(peg))}
    if len(peg) > 30:
        tip_z = np.percentile(peg[:, 2], 1)
        res["peg_tip_z"] = float(tip_z)
        res["peg_top_z"] = float(np.percentile(peg[:, 2], 99))
        res["peg_xy_raw"] = peg[:, :2].mean(0).tolist()
        body = peg[(peg[:, 2] > tip_z + 0.003) & (peg[:, 2] < tip_z + 0.025)]
        if len(body) > 30:
            res["peg_axis_xy"] = fit_circle_fixed_r(body[:, :2], 0.004, peg[:, :2].mean(0)).tolist()
    else:
        tip_z = None
    ceiling = (tip_z - 0.004) if tip_z is not None else tz + 0.1
    # candidate socket points: above table, below peg tip, near the hand in XY, not yellow
    cand = []
    for c, (P, y, ok, _) in data.items():
        m = ok & ~y & (P[..., 2] > tz + 0.004) & (P[..., 2] < ceiling)
        m &= np.linalg.norm(P[..., :2] - np.asarray(eef_xy)[None, None], axis=-1) < radius
        cand.append(P[m])
    cand = np.concatenate(cand)
    res["n_cand"] = int(len(cand))
    if len(cand) < 30:
        return res
    # socket top face: highest dense height cluster
    h, e = np.histogram(cand[:, 2], bins=np.arange(tz, ceiling + 0.001, 0.001))
    dense = np.nonzero(h >= max(20, 0.02 * len(cand)))[0]
    top = 0.5 * (e[dense[-1]] + e[dense[-1] + 1])
    sel = cand[np.abs(cand[:, 2] - top) < 0.0015]
    sel_z = np.median(sel[:, 2])
    xy = np.median(sel[:, :2], 0)
    for _ in range(3):  # trim outliers (other objects at same height)
        d = np.linalg.norm(sel[:, :2] - xy, axis=1)
        sel = sel[d < 0.012]
        xy = 0.5 * (sel[:, :2].min(0) + sel[:, :2].max(0))
    hole = hole_centre(sel, xy)
    if hole is not None:
        res["hole_xy"] = hole.tolist()
    res.update(socket_top_z=float(sel_z), socket_xy=xy.tolist(), socket_xy_mean=sel[:, :2].mean(0).tolist(),
               n_top=int(len(sel)), socket_extent=(sel[:, :2].max(0) - sel[:, :2].min(0)).tolist())
    return res


def fit_circle_fixed_r(xy, r, c0, iters=30):
    """Gauss-Newton fit of a circle centre with known radius to surface points."""
    c = np.asarray(c0, float).copy()
    for _ in range(iters):
        d = xy - c
        n = np.linalg.norm(d, axis=1) + 1e-9
        res = n - r
        w = 1.0 / (1.0 + (res / 0.001) ** 2)  # Cauchy robustness
        J = -d / n[:, None]
        A = (J * w[:, None]).T @ J; b = (J * w[:, None]).T @ res
        step = np.linalg.solve(A + 1e-9 * np.eye(2), -b)
        c += step
        if np.linalg.norm(step) < 1e-6:
            break
    return c


def hole_centre(top_pts, xy0, cell=0.0005, half=0.014):
    """Centroid of the empty region enclosed by the socket top face (the bore)."""
    from scipy import ndimage
    n = int(2 * half / cell)
    g = np.zeros((n, n), bool)
    ij = np.floor((top_pts[:, :2] - (np.asarray(xy0) - half)) / cell).astype(int)
    k = (ij >= 0).all(1) & (ij < n).all(1)
    g[ij[k, 0], ij[k, 1]] = True
    g = ndimage.binary_closing(g, iterations=1)
    lab, nl = ndimage.label(~g)
    best = None
    for L in range(1, nl + 1):
        m = lab == L
        ii, jj = np.nonzero(m)
        if ii.min() == 0 or jj.min() == 0 or ii.max() == n - 1 or jj.max() == n - 1:
            continue
        area = m.sum() * cell * cell
        if 2e-5 < area < 1.2e-4 and (best is None or area > best[0]):  # bore ~5e-5 m^2
            best = (area, np.array([ii.mean(), jj.mean()]))
    if best is None:
        return None
    return np.asarray(xy0) - half + (best[1] + 0.5) * cell
