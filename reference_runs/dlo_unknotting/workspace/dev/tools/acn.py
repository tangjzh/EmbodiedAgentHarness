"""Offline proxy: reconstruct the rope centreline from a capture and compute ACN / writhe (Gauss integrals).
usage: acn.py [plot=out.png] capdir...
Centreline = voxel (1 cm) shortest paths Qend->Q->P->tail end, smoothed. ACN uses the exact segment-pair solid angle."""
import heapq
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, '/workspace')
import rope_vision as rv

VOX = 0.01
OFFS = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]


def load(d):
    d = d.rstrip('/') + '/'
    obs = json.load(open(d + 'obs.json'))
    cams = {}
    for n in rv.CAMS:
        cams[n + '_rgb'] = np.load(d + n + '_rgb.npy')
        cams[n + '_depth_m'] = np.load(d + n + '_depth_m.npy')
    return obs, cams


def dijkstra(occ, s):
    dist, prev = {s: 0.0}, {s: None}
    pq = [(0.0, s)]
    while pq:
        dd, u = heapq.heappop(pq)
        if dd > dist[u]:
            continue
        for o in OFFS:
            v = (u[0] + o[0], u[1] + o[1], u[2] + o[2])
            if v in occ:
                nd = dd + VOX * np.sqrt(o[0] ** 2 + o[1] ** 2 + o[2] ** 2)
                if nd < dist.get(v, 1e9):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(pq, (nd, v))
    return dist, prev


def path(prev, t):
    out = []
    while t is not None:
        out.append(t)
        t = prev[t]
    return out[::-1]


def centreline(obs, cams):
    P = np.array(obs['robot']['right']['eef_position_world'])
    Q = np.array(obs['robot']['left']['eef_position_world'])
    pts, _ = rv.rope_points(obs, cams)
    keys = np.unique(np.floor(pts / VOX).astype(int), axis=0)
    occ = {tuple(k) for k in keys}
    cen = (keys + 0.5) * VOX
    near = lambda X: tuple(keys[np.argmin(np.linalg.norm(cen - X, axis=1))])
    sQ, sP = near(Q), near(P)
    dQ, prevQ = dijkstra(occ, sQ)
    dP, prevP = dijkstra(occ, sP)
    L = dQ.get(sP)
    # tail end: beyond P (dQ ~ dP + L), farthest from P; Q end: beyond Q (dP ~ dQ + L), farthest from Q
    tail = max((k for k in dP if k in dQ and dQ[k] - dP[k] > L - 0.04), key=lambda k: dP[k], default=sP)
    qend = max((k for k in dQ if k in dP and dP[k] - dQ[k] > L - 0.04), key=lambda k: dQ[k], default=sQ)
    seq = path(prevQ, qend)[::-1] + path(prevQ, sP)[1:] + path(prevP, tail)[1:]
    X = (np.array(seq) + 0.5) * VOX
    # smooth (moving average, keep ends) and resample at ~1 cm
    k = 3
    Xs = np.array([X[max(0, i - k):i + k + 1].mean(0) for i in range(len(X))])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Xs, axis=0), axis=1))]
    n = max(int(s[-1] / 0.01), 2)
    si = np.linspace(0, s[-1], n)
    Xr = np.stack([np.interp(si, s, Xs[:, j]) for j in range(3)], 1)
    info = dict(L_QP=L, L_tail=dP.get(tail), L_qend=dQ.get(qend), total=s[-1], P=P, Q=Q)
    return Xr, info


def solid_angle(p1, p2, p3, p4):
    """Signed Gauss integral contribution * 4pi of segments p1p2, p3p4 (Banchoff / Klenin-Langowski)."""
    r13, r14, r23, r24 = p3 - p1, p4 - p1, p3 - p2, p4 - p2
    def nrm(a, b):
        c = np.cross(a, b)
        n = np.linalg.norm(c)
        return c / n if n > 1e-12 else c
    n1, n2, n3, n4 = nrm(r13, r14), nrm(r14, r24), nrm(r24, r23), nrm(r23, r13)
    om = sum(np.arcsin(np.clip(np.dot(a, b), -1, 1)) for a, b in ((n1, n2), (n2, n3), (n3, n4), (n4, n1)))
    sgn = np.sign(np.dot(np.cross(p4 - p3, p2 - p1), r13))
    return om * sgn


def gauss(X, skip=1):
    n = len(X) - 1
    acn = wr = 0.0
    for i in range(n):
        for j in range(i + 1 + skip, n):
            om = solid_angle(X[i], X[i + 1], X[j], X[j + 1])
            acn += abs(om)
            wr += om
    return acn / (2 * np.pi), wr / (2 * np.pi)


def plot(Xs, names, out):
    ims = []
    for X, nm in zip(Xs, names):
        img = np.zeros((300, 600, 3), np.uint8)
        c = X.mean(0)
        f = lambda p: (int(300 + (p[1] - c[1]) * 600), int(150 - (p[0] - c[0]) * 600))
        for a, b in zip(X[:-1], X[1:]):
            col = tuple(int(v) for v in cv2.applyColorMap(np.uint8([[min(255, a[2] / 0.06 * 255)]]), cv2.COLORMAP_JET)[0, 0])
            cv2.line(img, f(a), f(b), col, 2)
        cv2.circle(img, f(X[0]), 4, (255, 255, 255), -1)
        cv2.putText(img, nm, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        ims.append(img)
    cv2.imwrite(out, np.concatenate(ims, 0))


if __name__ == '__main__':
    args = sys.argv[1:]
    out = None
    if args and args[0].startswith('plot='):
        out = args.pop(0)[5:]
    Xs, names = [], []
    for d in args:
        obs, cams = load(d)
        X, info = centreline(obs, cams)
        acn, wr = gauss(X)
        print('%-22s ACN %.3f  Wr %+.3f  L_QP %.3f L_tail %s L_qend %s total %.3f zmax %.3f' % (
            d, acn, wr, info['L_QP'], None if info['L_tail'] is None else round(info['L_tail'], 3),
            None if info['L_qend'] is None else round(info['L_qend'], 3), info['total'], X[:, 2].max()))
        Xs.append(X); names.append(d)
    if out:
        plot(Xs, names, out)
