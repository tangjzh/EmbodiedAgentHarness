"""Perception for PegInsertionSide from calibrated RGB-D (no object GT)."""
import numpy as np
import cv2
from geom import backproject, cam_mats


def cam_points(obs, cams, cam):
    K, E = cam_mats(obs, cam)
    rgb = cams[cam + '_rgb'].astype(np.int32)
    dep = cams[cam + '_depth_m'].astype(np.float64)
    valid = (dep > 0.01) & (dep < 5.0)
    P = backproject(np.where(valid, dep, 0.0), K, E)
    return rgb, dep, P, valid, K, E


def color_masks(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # peg head  #EC7357 (236,115,87): strong red, g/r ~ 0.49, b/r ~ 0.37
    orange = (r > 90) & (g < 0.68 * r) & (g > 0.3 * r) & (b < 0.55 * r) & (b > 0.18 * r)
    # peg tail  #EDF6F9: near white / slightly bluish, low saturation
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    white = (mx > 90) & ((mx - mn) < 0.16 * mx + 6) & (b >= r - 6)
    # box  #FFD289 (255,210,137): g/r ~ 0.82, b/r ~ 0.54
    box = (r > 60) & (g > 0.72 * r) & (g < 0.92 * r) & (b > 0.42 * r) & (b < 0.68 * r)
    return orange, white, box


def fit_peg_topdown(pts_xyz, is_orange, table_z=0.0):
    """Fit a flat-lying square peg (yaw only) to 3D points of the peg.
    Returns dict(center(x,y), yaw (head direction angle), L, r)."""
    z = pts_xyz[:, 2] - table_z
    ztop = np.percentile(z, 90)
    r = ztop / 2.0
    top = z > ztop - 0.006
    xy = pts_xyz[top, :2]
    c0 = xy.mean(0)
    C = np.cov((xy - c0).T)
    w, V = np.linalg.eigh(C)
    ax = V[:, 1]
    # head direction from orange vs white
    o = pts_xyz[is_orange, :2].mean(0) if is_orange.any() else None
    wpts = pts_xyz[~is_orange, :2]
    wm = wpts.mean(0) if len(wpts) else None
    if o is not None and wm is not None and np.dot(o - wm, ax) < 0:
        ax = -ax
    perp = np.array([-ax[1], ax[0]])
    s = (pts_xyz[:, :2] - c0) @ ax
    t = (pts_xyz[:, :2] - c0) @ perp
    smin, smax = np.percentile(s, 1), np.percentile(s, 99)
    L = (smax - smin) / 2.0
    center = c0 + ax * (smax + smin) / 2.0 + perp * np.median(t[top])
    return dict(center=center, axis=ax, yaw=float(np.arctan2(ax[1], ax[0])), L=float(L), r=float(r))


def detect_peg_base(obs, cams, table_z=None):
    rgb, dep, P, valid, K, E = cam_points(obs, cams, 'base_camera')
    z = P[..., 2]
    if table_z is None:
        tm = valid & (np.abs(z) < 0.006) & (np.abs(P[..., 0]) < 0.3) & (P[..., 1] > -0.45) & (P[..., 1] < 0.1)
        table_z = float(np.median(z[tm])) if tm.sum() > 50 else 0.0
    region = valid & (z > table_z + 0.006) & (z < 0.07) & (np.abs(P[..., 0]) < 0.3) & \
        (P[..., 1] > -0.45) & (P[..., 1] < 0.12)
    orange, white, box = color_masks(rgb)
    peg = region & (orange | white | (~box))
    # connected component around orange pixels
    n, lab = cv2.connectedComponents(peg.astype(np.uint8), connectivity=8)
    best, best_score = None, 0
    for i in range(1, n):
        m = lab == i
        sc = (m & orange).sum()
        if sc > best_score:
            best, best_score = m, sc
    if best is None:
        return None
    pts = P[best]
    fit = fit_peg_topdown(pts, orange[best], table_z)
    fit['table_z'] = table_z
    fit['npts'] = int(best.sum())
    fit['mask'] = best
    return fit


def detect_peg_hand(obs, cams, table_z=0.0, zmax=0.07):
    """Precise peg fit from the wrist camera looking down at a flat-lying peg."""
    rgb, dep, P, valid, K, E = cam_points(obs, cams, 'hand_camera')
    z = P[..., 2] - table_z
    orange, white, box = color_masks(rgb)
    cand = valid & (z > 0.006) & (z < zmax) & (orange | white)
    if cand.sum() < 30:
        return None
    n, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
    sizes = [(lab == i).sum() for i in range(n)]
    keep = np.zeros_like(cand)
    for i in range(1, n):
        if sizes[i] > 20:
            keep |= lab == i
    pts = P[keep]
    isor = orange[keep]
    zz = pts[:, 2] - table_z
    ztop = np.percentile(zz, 95)
    top = zz > ztop - 0.004
    xy = pts[top, :2]
    o_top = isor[top]
    c0 = xy.mean(0)
    w, V = np.linalg.eigh(np.cov((xy - c0).T))
    ax = V[:, 1]
    if o_top.any() and (~o_top).any():
        if np.dot(xy[o_top].mean(0) - xy[~o_top].mean(0), ax) < 0:
            ax = -ax
    perp = np.array([-ax[1], ax[0]])
    s = (xy - c0) @ ax
    t = (xy - c0) @ perp
    # width from robust extent across the axis
    tlo, thi = np.percentile(t, 1), np.percentile(t, 99)
    width = (thi - tlo) / 0.98          # the 1-99 percentile span of a uniform spread is 98% of it (ep11)
    tmid = 0.5 * (tlo + thi)
    # orange/white boundary along axis (peg centre)
    sb = None
    if o_top.sum() > 10 and (~o_top).sum() > 10:
        so, sw = s[o_top], s[~o_top]
        sb = 0.5 * (np.percentile(so, 2) + np.percentile(sw, 98))
    res = dict(axis=ax, yaw=float(np.arctan2(ax[1], ax[0])), r_top=float(ztop / 2.0), width=float(width),
               smin=float(s.min()), smax=float(s.max()), c0=c0, perp=perp, tmid=float(tmid), sb=sb,
               ntop=int(top.sum()), mask=keep)
    if sb is not None:
        res['center'] = c0 + ax * sb + perp * tmid
    return res


def _ransac_vertical_plane(pts, rng, iters=300, tol=0.002, yaw_ok=None):
    """RANSAC a vertical plane (horizontal normal). yaw_ok(n)->bool filters candidate normals."""
    best = (0, None, None)
    N = len(pts)
    if N < 30:
        return best
    for _ in range(iters):
        i, j = rng.choice(N, 2, replace=False)
        dxy = pts[j, :2] - pts[i, :2]
        nn = np.linalg.norm(dxy)
        if nn < 0.015:
            continue
        t = dxy / nn
        n = np.array([-t[1], t[0], 0.0])
        if yaw_ok is not None and not (yaw_ok(n) or yaw_ok(-n)):
            continue
        cnt = int((np.abs((pts - pts[i]) @ n) < tol).sum())
        if cnt > best[0]:
            best = (cnt, n, pts[i].copy())
    return best


def detect_hole(obs, cams, cam='hand_camera', h_half=None, prior_xy=None, prior_r=0.25, seed=0, min_behind=8):
    """Locate the box entrance face (vertical plane facing the robot side, normal ~ -y) and the square
    hole opening in it.  Hole pixels = valid pixels whose ray crosses the face plane inside the face's
    extent but whose surface lies > 3 mm behind the plane.
    Returns dict(n_out (outward face normal, horizontal), xb (= -n_out, insertion axis), face_pt,
    t (in-plane horizontal unit), hole centre (world), u/z extents, edge flags) or None."""
    rgb, dep, P, valid, K, E = cam_points(obs, cams, cam)
    orange, white, box = color_masks(rgb)
    z = P[..., 2]
    bm = box & valid & (z > 0.004)
    if prior_xy is not None:
        bm &= np.linalg.norm(P[..., :2] - np.asarray(prior_xy)[None, None], axis=-1) < prior_r
    pts = P[bm]
    if len(pts) < 60:
        return None
    cam_c = -E[:, :3].T @ E[:, 3]
    # outward normal of the entrance face points to -box_x, box yaw in [67.5,112.5] -> n_out yaw in [247.5,292.5]
    def yaw_ok(n):
        yw = np.degrees(np.arctan2(n[1], n[0])) % 360
        return 235 <= yw <= 305
    rng = np.random.default_rng(seed)
    cnt, n, p0 = _ransac_vertical_plane(pts, rng, yaw_ok=yaw_ok)
    if n is None or cnt < 40:
        return None
    inl = np.abs((pts - p0) @ n) < 0.002
    q = pts[inl]
    c = q.mean(0)
    A = q[:, :2] - c[:2]
    w, V = np.linalg.eigh(A.T @ A)
    t = V[:, 1]
    n = np.array([-t[1], t[0], 0.0])
    if np.dot(cam_c - c, n) < 0:
        n = -n                      # outward = toward camera
    if not yaw_ok(n):
        return None
    inl = np.abs((pts - c) @ n) < 0.002
    q = pts[inl]
    c = q.mean(0)
    c = c - n * np.dot(c - c, n)
    t = np.cross(np.array([0, 0, 1.0]), n)    # in-plane horizontal axis
    # face extent in (u,z) from inlier points
    uq = (q - c) @ t
    zq = q[:, 2]
    ulo, uhi = np.percentile(uq, 0.5), np.percentile(uq, 99.5)
    ztop = np.percentile(zq, 99.5)
    # ray / plane intersection for every pixel
    H, W = dep.shape
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_cam = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)
    d_w = d_cam @ E[:, :3]                     # R^T d
    denom = d_w @ n
    lam = np.where(np.abs(denom) > 1e-6, ((c - cam_c) @ n) / np.where(np.abs(denom) > 1e-6, denom, 1), np.nan)
    X = cam_c + lam[..., None] * d_w           # intersection with face plane
    xu = (X - c) @ t
    xz = X[..., 2]
    dist = (P - c) @ n                         # >0 in front (camera side)
    margin = 0.004
    inface = (lam > 0) & (xu > ulo + margin) & (xu < uhi - margin) & (xz > 0.004) & (xz < ztop - margin)
    behind = valid & inface & (dist < -0.003)
    onface = valid & inface & (np.abs(dist) < 0.0025)
    if behind.sum() < min_behind:
        return dict(n_out=n, xb=-n, face_pt=c, t=t, face_u=(ulo, uhi), face_ztop=ztop, hole=None,
                    nbehind=int(behind.sum()))
    # keep largest connected blob of behind pixels
    nlab, lab = cv2.connectedComponents(behind.astype(np.uint8), connectivity=4)
    sizes = np.bincount(lab[behind])
    k = int(np.argmax(sizes))
    hm = lab == k
    hu, hz = xu[hm], xz[hm]
    rows, cols = np.nonzero(hm)
    touches = dict(left=bool((cols == 0).any()), right=bool((cols == W - 1).any()),
                   top=bool((rows == 0).any()), bottom=bool((rows == H - 1).any()))
    # occlusion: opening bordered by something in front of the face (grasped peg, gripper)
    ring = cv2.dilate(hm.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool) & ~hm
    occl = ring & ((valid & (dist > 0.006)) | ~valid)
    touches['occl'] = bool(occl.sum() > 0.1 * max(1, ring.sum()))
    u0, u1 = hu.min(), hu.max()
    z0, z1 = hz.min(), hz.max()
    uc, zc = 0.5 * (u0 + u1), 0.5 * (z0 + z1)
    res = dict(n_out=n, xb=-n, face_pt=c, t=t, face_u=(ulo, uhi), face_ztop=ztop,
               u_ext=(float(u0), float(u1)), z_ext=(float(z0), float(z1)), npx=int(hm.sum()),
               touches=touches, mask=hm, onface=onface, uc=float(uc), zc=float(zc))
    res['hole'] = c + t * uc + np.array([0, 0, 1.0]) * (zc - c[2])
    res['hole'][2] = zc
    return res


def refine_hole(obs, cams, det, cam='hand_camera', h0=None, L=0.11, fit_h=True, roi=0.03, h_tol=None):
    """Model-based refinement of the hole opening: ray-trace a square tunnel (half-size h, axis -n_out)
    behind the fitted face plane and fit (u_c, z_c[, h]) to the observed depth-behind-plane of pixels
    around the opening.  Removes the threshold bias of detect_hole extents."""
    if det is None or det.get('hole') is None:
        return None
    rgb, dep, P, valid, K, E = cam_points(obs, cams, cam)
    n, c, t = det['n_out'], det['face_pt'], det['t']
    m = -n
    cam_c = -E[:, :3].T @ E[:, 3]
    H, W = dep.shape
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_cam = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)
    d_w = d_cam @ E[:, :3]
    denom = d_w @ n
    ok = denom < -1e-6                    # ray heading into the face
    lam = np.where(ok, ((c - cam_c) @ n) / np.where(ok, denom, -1), np.nan)
    X = cam_c + lam[..., None] * d_w
    ue = (X - c) @ t
    ze = X[..., 2]
    du, dz, dm = d_w @ t, d_w[..., 2], d_w @ m
    dist = (P - c) @ n
    u0, u1 = det['u_ext']; z0, z1 = det['z_ext']
    sel = ok & valid & (ue > u0 - roi) & (ue < u1 + roi) & (ze > z0 - roi) & (ze < z1 + roi) & (dist < 0.002)
    sel &= (ue > det['face_u'][0] + 0.003) & (ue < det['face_u'][1] - 0.003) & (ze < det['face_ztop'] - 0.003)
    ue, ze, du, dz, dm = ue[sel], ze[sel], du[sel], dz[sel], dm[sel]
    dobs = np.clip(-dist[sel], 0, 0.06)
    if len(dobs) < 50:
        return None
    eps = 1e-9
    big = 0.06

    def predict(uc, zc, h):
        inside = (np.abs(ue - uc) < h) & (np.abs(ze - zc) < h)
        su = np.where(du > eps, (uc + h - ue) / np.maximum(du, eps), np.where(du < -eps, (uc - h - ue) / np.minimum(du, -eps), np.inf))
        sz = np.where(dz > eps, (zc + h - ze) / np.maximum(dz, eps), np.where(dz < -eps, (zc - h - ze) / np.minimum(dz, -eps), np.inf))
        s = np.minimum(su, sz)
        se = 2 * L / np.maximum(dm, eps)
        d = np.where(s < se, dm * s, big)
        return np.where(inside, np.clip(d, 0, big), 0.0)

    def cost(uc, zc, h):
        r = predict(uc, zc, h) - dobs
        return np.sum(np.minimum(r * r, 0.004 ** 2))

    uc0 = det['uc']; zc0 = det['zc']
    hh0 = h0 if h0 is not None else 0.5 * max(u1 - u0, z1 - z0) + 0.001
    if h0 is None or (fit_h and h_tol is None):
        h_lo, h_hi = 0.017, 0.029            # hole half-size prior r+0.003 in [0.018,0.028]
    elif fit_h:
        h_lo, h_hi = h0 - h_tol, h0 + h_tol
    else:
        h_lo = h_hi = h0
    # start hypotheses: extent centre, and edge + h when the blob is cut by the image border
    # (a truncated view biases the extent centre toward the visible edge)
    touch = det.get('touches', {})
    starts = [(uc0, zc0)]
    hm = 0.5 * (h_lo + h_hi)
    if any(touch.values()):
        starts += [(uc0, z0 + hm), (uc0, z1 - hm), (u0 + hm, zc0), (u1 - hm, zc0),
                   (u0 + hm, z0 + hm), (u1 - hm, z0 + hm)]
    best = (np.inf, uc0, zc0, float(np.clip(hh0, h_lo, h_hi)))
    for (us, zs) in starts:
        for uc in np.arange(us - 0.012, us + 0.012 + 1e-9, 0.002):
            for zc in np.arange(zs - 0.012, zs + 0.012 + 1e-9, 0.002):
                for h in np.arange(h_lo, h_hi + 1e-9, 0.002) if h_hi > h_lo else [h_lo]:
                    cc = cost(uc, zc, h)
                    if cc < best[0]:
                        best = (cc, uc, zc, h)
    for (span, stp) in [(0.003, 0.0005), (0.0006, 0.0001)]:
        _, ub, zb, hb = best
        hs = np.arange(max(h_lo, hb - span / 2), min(h_hi, hb + span / 2) + 1e-9, stp) if h_hi > h_lo else [h_lo]
        for uc in np.arange(ub - span, ub + span + 1e-9, stp):
            for zc in np.arange(zb - span, zb + span + 1e-9, stp):
                for h in hs:
                    cc = cost(uc, zc, h)
                    if cc < best[0]:
                        best = (cc, uc, zc, h)
    cst, uc, zc, h = best
    hole = c + t * uc
    hole[2] = zc
    pred = predict(uc, zc, h)
    agree = float(np.mean(np.abs(pred - dobs) < 0.003))
    return dict(hole=hole, uc=float(uc), zc=float(zc), h=float(h), cost=float(cst), npx=int(len(dobs)), agree=agree,
                xb=-n, n_out=n, face_pt=c, t=t)


def peg_in_hand(obs, cams, p_tcp, R_tcp, r_est=0.0225, L_est=0.11):
    """Estimate the grasped peg's pose in the TCP frame (x fwd ~ head, y finger axis, z down) from the
    wrist camera, which sees the peg's top face for TCP x in ~[0.0, 0.09].
    Returns dict(theta = pitch (head-down +), psi = yaw (about TCP z), z_top0 = top-face z at x=0,
    y0 = centre-line y at x=0, xb = orange/white boundary x (peg centre), head = head-centre (TCP frame),
    axis = peg axis (TCP frame), n = number of top points)."""
    rgb, dep, P, valid, K, E = cam_points(obs, cams, 'hand_camera')
    orange, white, box = color_masks(rgb)
    loc = (P - p_tcp) @ R_tcp
    x, y, z = loc[..., 0], loc[..., 1], loc[..., 2]
    ywin = max(r_est - 0.005, 0.008)
    cand = valid & (orange | white) & (x > 0.008) & (x < 0.2) & (np.abs(y) < ywin) & (z > -r_est - 0.03) & (z < -r_est + 0.03)
    if cand.sum() < 80:
        return None
    X = np.stack([x[cand], y[cand], z[cand]], 1)
    isor = orange[cand]
    # robust plane z = a + b x + c y (top face)
    A = np.c_[np.ones(len(X)), X[:, 0], X[:, 1]]
    w = np.ones(len(X), bool)
    for it in range(4):
        coef, *_ = np.linalg.lstsq(A[w], X[w, 2], rcond=None)
        res = X[:, 2] - A @ coef
        w = np.abs(res) < (0.004 if it < 2 else 0.0018)
        if w.sum() < 50:
            return None
    a, b, c = coef
    theta = float(np.arctan(b))
    T = X[w]; To = isor[w]
    # centre line / yaw from all top-face pixels of the wider window
    cand2 = valid & (orange | white) & (x > 0.008) & (x < 0.2) & (np.abs(y) < r_est + 0.006)
    X2 = np.stack([x[cand2], y[cand2], z[cand2]], 1)
    on_top = np.abs(X2[:, 2] - (a + b * X2[:, 0] + c * X2[:, 1])) < 0.002
    X2 = X2[on_top]
    psi, y0, width = 0.0, 0.0, np.nan
    if len(X2) > 80:
        bins = np.arange(X2[:, 0].min(), X2[:, 0].max(), 0.005)
        mids, cents, wids = [], [], []
        for b0 in bins:
            s = (X2[:, 0] >= b0) & (X2[:, 0] < b0 + 0.005)
            if s.sum() > 15:
                lo, hi = np.percentile(X2[s, 1], 1), np.percentile(X2[s, 1], 99)
                mids.append(b0 + 0.0025); cents.append(0.5 * (lo + hi)); wids.append(hi - lo)
        if len(mids) >= 4:
            pc = np.polyfit(mids, cents, 1)
            psi, y0 = float(np.arctan(pc[0])), float(pc[1])
            width = float(np.median(wids))
    xb = None
    if To.sum() > 20 and (~To).sum() > 20:
        xo, xw = T[To, 0], T[~To, 0]
        if np.median(xo) > np.median(xw):
            xb = float(0.5 * (np.percentile(xo, 1) + np.percentile(xw, 99)))
    elif To.sum() > 20 and (~To).sum() <= 20:
        xb = float(np.percentile(T[To, 0], 1))   # only orange visible: boundary at/behind the first orange
    ax = np.array([np.cos(theta) * np.cos(psi), np.sin(psi) * np.cos(theta), np.sin(theta)])
    ax /= np.linalg.norm(ax)
    res = dict(theta=theta, psi=psi, z_top0=float(a), roll_slope=float(c), y0=y0, width=width, xb=xb,
               n=int(w.sum()), axis=ax, n_orange=int(To.sum()), n_white=int((~To).sum()))
    if xb is not None:
        n_up = np.array([np.sin(theta), 0.0, -np.cos(theta)])
        q_b = np.array([xb, y0 + np.tan(psi) * xb, a + b * xb])
        centre = q_b - r_est * n_up
        res['centre'] = centre
        res['head'] = centre + L_est * ax
    return res
