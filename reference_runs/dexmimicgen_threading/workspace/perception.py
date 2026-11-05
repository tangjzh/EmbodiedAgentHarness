"""Perception for TwoArmThreading from calibrated RGB-D only (no object state).

agentview_objects : coarse needle (handle cube + shaft) and tripod-ring poses from the fixed agentview camera
refine_ring       : ring top-bar rectangle from robot0's wrist camera looking down
refine_cube       : needle handle top-face square from robot1's wrist camera looking down
measure_inhand    : ring hole + needle shaft line in robot0's hand frame (robot0 holds the ring, hand x = ring axis)
"""
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as Rot

from geom import deproject, cam_mats

# object geometry (measured in episode 1)
CUBE = 0.040          # needle handle cube edge
SHAFT_HALF = 0.005    # shaft half width (10 mm square)
RING_HOLE_HALF = 0.007
RING_TOP_H = 0.196    # ring top above the table (tripod standing)


def cloud(obs, cams, cam):
    K, T = cam_mats(obs, cam)
    return deproject(cams[cam + '_depth_m'], K, T)


def min_area_rect(xy, step_deg=0.25, period=90.0):
    """Min-area bounding rectangle over angles in [0, period). Returns (yaw, centre, extents) with
    extents[0] along (cos yaw, sin yaw)."""
    best = None
    for a in np.radians(np.arange(0, period, step_deg)):
        R2 = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        r = xy @ R2.T
        lo, hi = np.percentile(r, 0.5, 0), np.percentile(r, 99.5, 0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            best = (area, a, lo, hi, R2)
    _, a, lo, hi, R2 = best
    return a, R2.T @ ((lo + hi) / 2), hi - lo


def table_height(P):
    z = P[..., 2]
    m = (P[..., 0] > -0.3) & (z > 0.6) & (z < 1.0)
    h, e = np.histogram(z[m], bins=np.arange(0.6, 1.0, 0.001))
    return e[h.argmax()] + 0.0005


def agentview_objects(obs, cams):
    """Coarse object poses from agentview. Returns dict(table_z, needle=..., ring=...)."""
    P = cloud(obs, cams, 'agentview')
    rgb = cams['agentview_rgb'].astype(float)
    zt = table_height(P)
    z = P[..., 2]
    cand = (z > zt + 0.004) & (z < zt + 0.215) & (P[..., 0] > -0.35) & (np.abs(P[..., 1]) < 0.45)
    for i in range(2):
        e = np.array(obs['robot'][f'robot{i}_eef_pos'])
        cand &= ~((np.hypot(P[..., 0] - e[0], P[..., 1] - e[1]) < 0.07) & (z > e[2] - 0.04))
    p, c = P[cand], rgb[cand]
    res = 0.005
    ij = np.floor((p[:, :2] - [-0.4, -0.5]) / res).astype(int)
    G = np.zeros((int(0.8 / res) + 2, int(1.0 / res) + 2), bool)
    G[ij[:, 0], ij[:, 1]] = True
    lab, n = ndimage.label(ndimage.binary_closing(G, iterations=2) | G)
    pl = lab[ij[:, 0], ij[:, 1]]
    comps = []
    for k in range(1, n + 1):
        s = pl == k
        if s.sum() < 15:
            continue
        q = p[s]
        comps.append(dict(k=k, n=int(s.sum()), hmax=q[:, 2].max() - zt, xy=q[:, :2].mean(0), br=c[s].mean()))
    out = dict(table_z=zt)
    # tripod: component whose top is closest to the ring-top height (robot parts entering the band are clipped at
    # its 0.215 m top); fall back to the tallest
    tall = [cc for cc in comps if cc['hmax'] > 0.12 and cc['n'] >= 40]
    tri = min(tall, key=lambda cc: abs(cc['hmax'] - RING_TOP_H)) if tall else max(comps, key=lambda cc: cc['hmax'])
    q = p[pl == tri['k']]
    hq = q[:, 2] - zt
    ring = q[hq > hq.max() - 0.03]
    plate = q[(hq > 0.07) & (hq < 0.10)]
    a_pl = None
    if len(plate) > 50:
        a_pl, _, _ = min_area_rect(plate[:, :2], 0.5)
    _, s_, vt = np.linalg.svd(ring[:, :2] - ring[:, :2].mean(0))
    ax = vt[1]
    a_ring = np.arctan2(ax[1], ax[0]) % np.pi
    if a_pl is not None:   # ring axis is parallel to a plate edge; pick the edge closest to the ring PCA minor axis
        cands = [(a_pl + k * np.pi / 2) % np.pi for k in range(2)]
        a_ring = min(cands, key=lambda a: abs(np.angle(np.exp(2j * (a - a_ring)))))
    out['ring'] = dict(center=np.array([*ring[:, :2].mean(0), q[:, 2].max() - 0.012]), top_z=q[:, 2].max(),
                       axis_yaw=a_ring, n=len(ring), plate_yaw=a_pl)
    # needle: dark, low components near the biggest dark low one
    nd = [cc for cc in comps if cc['hmax'] < 0.07 and cc['k'] != tri['k']]
    if nd:
        big = max(nd, key=lambda cc: cc['n'])
        nd = [cc for cc in nd if np.linalg.norm(cc['xy'] - big['xy']) < 0.2]
        q = p[np.isin(pl, [cc['k'] for cc in nd])]
        hq = q[:, 2] - zt
        top = q[hq > 0.032]
        a, ctr, ext = min_area_rect(top[:, :2])
        sh = q[(hq < 0.03) & (np.linalg.norm(q[:, :2] - ctr, axis=1) > 0.035)]
        if len(sh) > 5:
            v = sh[:, :2].mean(0) - ctr
            phi = np.arctan2(v[1], v[0])
        else:
            phi = np.nan
        out['needle'] = dict(center=np.array([ctr[0], ctr[1], zt + CUBE / 2]), top_z=np.median(top[:, 2]),
                             face_yaw=a, shaft_yaw=phi, ext=ext, n_top=len(top), n_shaft=len(sh))
    return out


def refine_ring(obs, cams, c0, cam='robot0_eye_in_hand', win=0.04):
    """Ring top bar (24 x 10 mm rectangle) seen from above. c0: coarse ring centre (world)."""
    P = cloud(obs, cams, cam)
    w = (np.hypot(P[..., 0] - c0[0], P[..., 1] - c0[1]) < win) & (np.abs(P[..., 2] - c0[2]) < 0.04)
    q = P[w]
    if len(q) < 30:
        return None
    zt = np.percentile(q[:, 2], 99.5)
    top = q[q[:, 2] > zt - 0.003]
    ctr = top[:, :2].mean(0)
    _, s, vt = np.linalg.svd(top[:, :2] - ctr)
    r = (top[:, :2] - ctr) @ vt.T
    ext = np.percentile(r, 99, 0) - np.percentile(r, 1, 0)
    ax = vt[1]
    return dict(center=np.array([ctr[0], ctr[1], zt - 0.012]), top_z=zt, axis_yaw=np.arctan2(ax[1], ax[0]) % np.pi,
                ext=ext, n=len(top))


def refine_cube(obs, cams, c0, table_z, cam='robot1_eye_in_hand', win=0.05):
    """Needle handle top face (40 mm square) from above. c0: coarse cube centre (world)."""
    P = cloud(obs, cams, cam)
    w = (np.hypot(P[..., 0] - c0[0], P[..., 1] - c0[1]) < win) & (P[..., 2] > table_z + 0.032) & \
        (P[..., 2] < table_z + 0.05)
    top = P[w]
    if len(top) < 50:
        return None
    a, ctr, ext = min_area_rect(top[:, :2])
    return dict(center=np.array([ctr[0], ctr[1], np.median(top[:, 2]) - CUBE / 2]), face_yaw=a, ext=ext, n=len(top))


# ------------------------------------------------------------------ in-hand measurement (robot0 wrist)
def hand_points(obs, cams, cam='robot0_eye_in_hand', arm=0):
    """Deproject a camera into the given arm's hand frame. Returns (Ph HxWx3, P world HxWx3)."""
    P = cloud(obs, cams, cam)
    e = np.array(obs['robot'][f'robot{arm}_eef_pos'])
    R = Rot.from_quat(obs['robot'][f'robot{arm}_eef_quat']).as_matrix()
    return (P - e) @ R, P


def measure_ring(Ph):
    """Ring front face plane (hand x) and hole bottom edge (hand z, down positive) in robot0's hand frame."""
    x, y, z = Ph[..., 0], Ph[..., 1], Ph[..., 2]
    m = (np.abs(y) < 0.015) & (z > -0.012) & (z < 0.012) & (x > -0.002) & (x < 0.012)
    p = Ph[m]
    face_x = np.median(p[:, 0]) if len(p) else np.nan
    mb = (np.abs(y) < 0.006) & (x < face_x - 0.0005) & (x > face_x - 0.012) & (z > 0.002) & (z < 0.013)
    pb = Ph[mb]
    bottom_z = np.median(pb[:, 2]) if len(pb) > 5 else np.nan
    return dict(face_x=face_x, bottom_z=bottom_z, hole_z=bottom_z - RING_HOLE_HALF, n_face=len(p), n_bot=len(pb))


def robust_line(x, v, iters=3, thr=0.0008):
    keep = np.ones(len(x), bool)
    for _ in range(iters):
        A = np.stack([np.ones(keep.sum()), x[keep]], 1)
        c = np.linalg.lstsq(A, v[keep], rcond=None)[0]
        r = np.abs(v - (c[0] + c[1] * x))
        keep = r < max(thr, 2.5 * np.median(r[keep]))
    return c, keep


def measure_shaft(Ph, face_x, xmax=0.09):
    """Needle shaft seen from above in robot0's hand frame (hand x = ring axis toward the needle, y lateral, z down).
    Returns dict(tip_x, y0, sy, z0, sz) with lateral centre y = y0 + sy*x and axis z = z0 + sz*x, or None."""
    x, y, z = Ph[..., 0], Ph[..., 1], Ph[..., 2]
    m = (x > face_x + 0.002) & (x < xmax) & (np.abs(y) < 0.016) & (z > -0.025) & (z < 0.02)
    # robot0's own fingers sit at |y| >~ 12 mm next to the ring
    m &= ~((x < face_x + 0.012) & (np.abs(y) > 0.0085))
    p = Ph[m]
    if len(p) < 40:
        return None
    rows = []
    for x0 in np.arange(p[:, 0].min(), p[:, 0].max(), 0.002):
        s = p[(p[:, 0] >= x0) & (p[:, 0] < x0 + 0.002)]
        if len(s) < 4:
            continue
        top = s[:, 2].min()
        st = s[s[:, 2] < top + 0.0008]
        rows.append((x0 + 0.001, top, st[:, 1].min(), st[:, 1].max(), len(s)))
    rows = np.array(rows)
    if len(rows) < 5:
        return None
    # contiguous run of bins starting at the ring side (the shaft) with a plausible width
    wid = rows[:, 3] - rows[:, 2]
    ok = (wid > 0.006) & (wid < 0.012)
    use = rows[ok]
    if len(use) < 4:
        return None
    tip_x = use[0, 0] - 0.001
    use = use[use[:, 0] > tip_x + 0.004]  # skip the end face
    if len(use) < 3:
        return None
    (z0, sz), kz = robust_line(use[:, 0], use[:, 1] + SHAFT_HALF)
    (y0, sy), ky = robust_line(use[:, 0], (use[:, 2] + use[:, 3]) / 2)
    return dict(tip_x=tip_x, y0=y0, sy=sy, z0=z0, sz=sz, width=float(np.median(use[:, 3] - use[:, 2])),
                n=int(m.sum()), nbins=len(use), x_range=(use[0, 0], use[-1, 0]))


def shaft_votes(P, c0, table_z, face_yaw, rmin=0.025, rmax=0.15):
    """Needle-shaft evidence along the four handle face normals (face_yaw + k*90deg): number of points at shaft
    height (8-32 mm above the table) within 1 cm of each candidate ray from the handle centre c0.
    Returns (yaws, counts)."""
    h = P[..., 2] - table_z
    d = P[..., :2] - np.asarray(c0)[:2]
    r = np.hypot(d[..., 0], d[..., 1])
    q = d[(h > 0.008) & (h < 0.032) & (r > rmin) & (r < rmax)]
    yaws = face_yaw + np.arange(4) * np.pi / 2
    counts = []
    for a in yaws:
        u = np.array([np.cos(a), np.sin(a)])
        counts.append(int(np.sum((q @ u > rmin) & (np.abs(q @ [-u[1], u[0]]) < 0.01))))
    return yaws, np.array(counts)
