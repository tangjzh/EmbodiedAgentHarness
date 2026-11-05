"""Offline proxy v2: rope centreline from a fine top-down rope mask (medial shortest path between the two rope ends,
gripper footprints forced into the mask) with heights from the top-surface map; Gauss-integral ACN / writhe.
usage: acn2.py [plot=out.png] capdir..."""
import json
import sys

import cv2
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, '/workspace')
import rope_vision as rv  # noqa: E402

RES = 0.003
R_ROPE = 0.0075


def load(d):
    d = d.rstrip('/') + '/'
    obs = json.load(open(d + 'obs.json'))
    cams = {}
    for n in rv.CAMS:
        cams[n + '_rgb'] = np.load(d + n + '_rgb.npy')
        cams[n + '_depth_m'] = np.load(d + n + '_depth_m.npy')
    return obs, cams


def grid_graph(mask, cost):
    H, W = mask.shape
    idx = -np.ones((H, W), int)
    rr, cc = np.nonzero(mask)
    idx[rr, cc] = np.arange(len(rr))
    rows, cols, w = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r2, c2 = rr + dr, cc + dc
        ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
        ok[ok] &= mask[r2[ok], c2[ok]]
        a, b = idx[rr[ok], cc[ok]], idx[r2[ok], c2[ok]]
        step = np.hypot(dr, dc)
        ww = step * 0.5 * (cost[rr[ok], cc[ok]] + cost[r2[ok], c2[ok]])
        rows += [a, b]; cols += [b, a]; w += [ww, ww]
    n = len(rr)
    G = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    return G, rr, cc


def centreline(obs, cams):
    P = np.array(obs['robot']['right']['eef_position_world'])
    Q = np.array(obs['robot']['left']['eef_position_world'])
    pts, _ = rv.rope_points(obs, cams, zmax=0.6)
    x0, x1 = min(P[0], Q[0]) - 0.3, max(P[0], Q[0]) + 0.3
    y0, y1 = min(P[1], Q[1]) - 0.3, max(P[1], Q[1]) + 0.4
    zb = rv.height_map(pts, x0, x1, y0, y1, RES)
    occ = zb > -0.5
    for E in (P, Q):  # gripper footprint (rope hidden under the fingers)
        r, c = int((x1 - E[0]) / RES), int((E[1] - y0) / RES)
        cv2.circle(occ.view(np.uint8), (c, r), int(0.012 / RES), 1, -1)
        zb[max(0, r - 4):r + 5, max(0, c - 4):c + 5] = np.maximum(zb[max(0, r - 4):r + 5, max(0, c - 4):c + 5], E[2] + R_ROPE)
    occ = ndimage.binary_closing(occ, np.ones((3, 3)), iterations=2)
    lab, nl = ndimage.label(occ, np.ones((3, 3)))
    rP, cP = int((x1 - P[0]) / RES), int((P[1] - y0) / RES)
    occ = lab == lab[rP, cP]
    dt = ndimage.distance_transform_edt(occ)
    cost = 1.0 / (dt + 0.5) ** 2
    G, rr, cc = grid_graph(occ, cost)
    iP = np.argmin((rr - rP) ** 2 + (cc - cP) ** 2)
    # rope ends: geodesically farthest point from P, then farthest from that one
    d0 = dijkstra(G, indices=iP, unweighted=False)
    geo = dijkstra(G, indices=iP, unweighted=True)
    e1 = np.argmax(np.where(np.isfinite(geo), geo, -1))
    d1, pred = dijkstra(G, indices=e1, return_predecessors=True, unweighted=True)
    e2 = np.argmax(np.where(np.isfinite(d1), d1, -1))
    dm, pred = dijkstra(G, indices=e1, return_predecessors=True)
    seq = [e2]
    while pred[seq[-1]] >= 0:
        seq.append(pred[seq[-1]])
    seq = np.array(seq[::-1])
    xy = np.stack([x1 - (rr[seq] + 0.5) * RES, y0 + (cc[seq] + 0.5) * RES], 1)
    # heights: local max of the top-surface map minus the rope radius
    zl = ndimage.maximum_filter(np.where(zb > -0.5, zb, 0), size=3)
    z = np.maximum(zl[rr[seq], cc[seq]] - R_ROPE, R_ROPE)
    X = np.column_stack([xy, z])
    # smooth lightly and resample at 1 cm
    k = 2
    Xs = np.array([X[max(0, i - k):i + k + 1].mean(0) for i in range(len(X))])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Xs, axis=0), axis=1))]
    n = max(int(s[-1] / 0.01), 2)
    si = np.linspace(0, s[-1], n)
    Xr = np.stack([np.interp(si, s, Xs[:, j]) for j in range(3)], 1)
    # orient: start at the end nearer to Q
    if np.linalg.norm(Xr[0, :2] - Q[:2]) > np.linalg.norm(Xr[-1, :2] - Q[:2]):
        Xr = Xr[::-1]
    return Xr, dict(P=P, Q=Q, total=s[-1])


def solid_angle(p1, p2, p3, p4):
    r13, r14, r23, r24 = p3 - p1, p4 - p1, p3 - p2, p4 - p2

    def nrm(a, b):
        c = np.cross(a, b)
        n = np.linalg.norm(c)
        return c / n if n > 1e-12 else c
    n1, n2, n3, n4 = nrm(r13, r14), nrm(r14, r24), nrm(r24, r23), nrm(r23, r13)
    om = sum(np.arcsin(np.clip(np.dot(a, b), -1, 1)) for a, b in ((n1, n2), (n2, n3), (n3, n4), (n4, n1)))
    return om * np.sign(np.dot(np.cross(p4 - p3, p2 - p1), r13))


def gauss(X):
    n = len(X) - 1
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 2, n):
            M[i, j] = M[j, i] = solid_angle(X[i], X[i + 1], X[j], X[j + 1])
    return np.abs(M).sum() / 2 / (2 * np.pi), M.sum() / 2 / (2 * np.pi), np.abs(M).sum(1) / (4 * np.pi)


if __name__ == '__main__':
    args = sys.argv[1:]
    out = None
    if args and args[0].startswith('plot='):
        out = args.pop(0)[5:]
    ims = []
    for d in args:
        obs, cams = load(d)
        X, info = centreline(obs, cams)
        acn, wr, per = gauss(X)
        iP = np.argmin(np.linalg.norm(X[:, :2] - info['P'][:2], axis=1))
        sP = iP * 0.01
        # contributions by region along the rope (arc length relative to P)
        reg = {'S far': (-9, -0.25), 'S near P': (-0.25, -0.03), 'at P': (-0.03, 0.03), 'tail': (0.03, 9)}
        parts = {k: per[[i for i in range(len(per)) if a <= i * 0.01 - sP < b]].sum() for k, (a, b) in reg.items()}
        print('%-24s ACN %.3f Wr %+.3f  len %.3f  sP %.2f  zmax %.3f  | %s' % (
            d, acn, wr, info['total'], sP, X[:, 2].max(), '  '.join('%s %.3f' % kv for kv in parts.items())))
        if out:
            img = np.zeros((260, 900, 3), np.uint8)
            c = info['P']
            f = lambda p: (int(600 + (p[1] - c[1]) * 800), int(130 - (p[0] - c[0]) * 800))
            for a, b in zip(X[:-1], X[1:]):
                col = tuple(int(v) for v in cv2.applyColorMap(np.uint8([[min(255, a[2] / 0.05 * 255)]]), cv2.COLORMAP_JET)[0, 0])
                cv2.line(img, f(a), f(b), col, 2)
            cv2.circle(img, f(X[0]), 4, (255, 255, 255), -1)
            cv2.drawMarker(img, f(info['P']), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
            cv2.putText(img, '%s ACN %.3f' % (d, acn), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            ims.append(img)
    if out:
        cv2.imwrite(out, np.concatenate(ims, 0))
