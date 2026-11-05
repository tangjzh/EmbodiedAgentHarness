"""Offline proxy v3 for hanging (non-flat) rope states: 3-D voxel centreline (fused rope points of all cameras,
dilated into a solid tube, gripper TCPs forced in, medial shortest path between the geodesic-farthest ends),
then the Gauss-integral ACN / writhe as in acn2.  usage: acn3d.py capdir..."""
import sys

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/dev/tools')
import acn2  # noqa: E402
import rope_vision as rv  # noqa: E402

VOX = 0.004
R_ROPE = 0.0075


def centreline3d(obs, cams, dil=2):
    P = np.array(obs['robot']['right']['eef_position_world'])
    Q = np.array(obs['robot']['left']['eef_position_world'])
    pts, _ = rv.rope_points(obs, cams, zmax=0.7)
    pts = pts[pts[:, 2] > 0.002]
    lo = np.minimum(pts.min(0), np.minimum(P, Q)) - 0.03
    hi = np.maximum(pts.max(0), np.maximum(P, Q)) + 0.03
    shape = np.ceil((hi - lo) / VOX).astype(int) + 1
    occ = np.zeros(shape, bool)
    ijk = ((pts - lo) / VOX).astype(int)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    occ = ndimage.binary_dilation(occ, ndimage.generate_binary_structure(3, 1), iterations=dil)
    for E in (P, Q):  # rope hidden in the fingers: force a small ball at the TCP
        c = ((E - lo) / VOX).astype(int)
        rr = int(0.012 / VOX)
        sl = tuple(slice(max(0, c[i] - rr), c[i] + rr + 1) for i in range(3))
        g = np.stack(np.meshgrid(*[np.arange(s.start, s.stop) for s in sl], indexing='ij'), -1)
        occ[sl] |= (np.linalg.norm(g - c, axis=-1) <= rr)
    lab, nl = ndimage.label(occ, np.ones((3, 3, 3)))
    cP = tuple(((P - lo) / VOX).astype(int))
    occ = lab == lab[cP]
    dt = ndimage.distance_transform_edt(occ)
    cost = 1.0 / (dt + 0.5) ** 2
    idx = -np.ones(occ.shape, int)
    vv = np.argwhere(occ)
    idx[tuple(vv.T)] = np.arange(len(vv))
    rows, cols, w = [], [], []
    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) > (0, 0, 0)]
    for o in offs:
        v2 = vv + o
        ok = np.all((v2 >= 0) & (v2 < occ.shape), 1)
        ok[ok] &= occ[tuple(v2[ok].T)]
        a, b = idx[tuple(vv[ok].T)], idx[tuple(v2[ok].T)]
        ww = np.linalg.norm(o) * 0.5 * (cost[tuple(vv[ok].T)] + cost[tuple(v2[ok].T)])
        rows += [a, b]; cols += [b, a]; w += [ww, ww]
    n = len(vv)
    G = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    iP = idx[cP]
    geo = dijkstra(G, indices=iP, unweighted=True)
    e1 = np.argmax(np.where(np.isfinite(geo), geo, -1))
    d1 = dijkstra(G, indices=e1, unweighted=True)
    e2 = np.argmax(np.where(np.isfinite(d1), d1, -1))
    dm, pred = dijkstra(G, indices=e1, return_predecessors=True)
    seq = [e2]
    while pred[seq[-1]] >= 0:
        seq.append(pred[seq[-1]])
    X = lo + (vv[np.array(seq[::-1])] + 0.5) * VOX
    # floor-lying rope: only its top surface is visible -> centre at most top - r
    X[:, 2] = np.where(X[:, 2] < 0.03, np.maximum(X[:, 2] - R_ROPE, R_ROPE), X[:, 2])
    k = 2
    Xs = np.array([X[max(0, i - k):i + k + 1].mean(0) for i in range(len(X))])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Xs, axis=0), axis=1))]
    m = max(int(s[-1] / 0.01), 2)
    si = np.linspace(0, s[-1], m)
    Xr = np.stack([np.interp(si, s, Xs[:, j]) for j in range(3)], 1)
    if np.linalg.norm(Xr[0] - Q) > np.linalg.norm(Xr[-1] - Q):
        Xr = Xr[::-1]
    return Xr, dict(P=P, Q=Q, total=s[-1])


if __name__ == '__main__':
    for d in sys.argv[1:]:
        obs, cams = acn2.load(d)
        X, info = centreline3d(obs, cams)
        acn, wr, per = acn2.gauss(X)
        iP = np.argmin(np.linalg.norm(X - info['P'], axis=1))
        sP = iP * 0.01
        reg = {'S far': (-9, -0.25), 'S near P': (-0.25, -0.03), 'at P': (-0.03, 0.03), 'tail': (0.03, 9)}
        parts = {k: per[[i for i in range(len(per)) if a <= i * 0.01 - sP < b]].sum() for k, (a, b) in reg.items()}
        print('%-26s ACN3 %.3f Wr %+.3f len %.3f sP %.2f Pz %.3f dP %.3f | %s' % (
            d, acn, wr, info['total'], sP, info['P'][2], np.linalg.norm(X[iP] - info['P']),
            '  '.join('%s %.3f' % kv for kv in parts.items())))
