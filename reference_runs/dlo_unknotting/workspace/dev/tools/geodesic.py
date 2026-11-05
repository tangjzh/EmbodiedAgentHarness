"""Offline: approx rope arc length between the two grippers via shortest path over occupied voxels.
usage: geodesic.py capdir... (voxel 1 cm, 26-connectivity)"""
import heapq
import json
import sys

import numpy as np

sys.path.insert(0, '/workspace')
import rope_vision as rv

VOX = 0.01


def geo(d):
    d = d.rstrip('/') + '/'
    obs = json.load(open(d + 'obs.json'))
    cams = {}
    for n in rv.CAMS:
        cams[n + '_rgb'] = np.load(d + n + '_rgb.npy')
        cams[n + '_depth_m'] = np.load(d + n + '_depth_m.npy')
    P = np.array(obs['robot']['right']['eef_position_world'])
    Q = np.array(obs['robot']['left']['eef_position_world'])
    pts, ci = rv.rope_points(obs, cams)
    keys = np.unique(np.floor(pts / VOX).astype(int), axis=0)
    occ = {tuple(k) for k in keys}
    def nearest(X):
        k = keys[np.argmin(np.linalg.norm((keys + 0.5) * VOX - X, axis=1))]
        return tuple(k)
    s, t = nearest(Q), nearest(P)
    dist = {s: 0.0}
    pq = [(0.0, s)]
    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]
    while pq:
        dd, u = heapq.heappop(pq)
        if u == t:
            break
        if dd > dist.get(u, 1e9):
            continue
        for o in offs:
            v = (u[0] + o[0], u[1] + o[1], u[2] + o[2])
            if v in occ:
                nd = dd + VOX * np.sqrt(o[0] ** 2 + o[1] ** 2 + o[2] ** 2)
                if nd < dist.get(v, 1e9):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
    sk = (np.array(s) + 0.5) * VOX; tk = (np.array(t) + 0.5) * VOX
    return obs.get('native_step'), dist.get(t), np.linalg.norm(sk - Q), np.linalg.norm(tk - P), np.linalg.norm(P - Q)


for d in sys.argv[1:]:
    st, g, eq, ep, pq = geo(d)
    print('%s step %s geodesic Q->P %s  (snap Q %.3f P %.3f)  |P-Q| %.3f' % (d, st, None if g is None else round(g, 3), eq, ep, pq))
