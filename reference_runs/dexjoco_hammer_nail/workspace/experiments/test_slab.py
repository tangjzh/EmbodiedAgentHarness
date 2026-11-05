"""Offline: face slab extents per camera on E1 carry observations."""
import os
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.insert(0, 'code/hammer')
sys.path.insert(0, 'experiments')
import numpy as np
import perception as pc
from test_percep_lib import load_saved

nail_z = 1.1227
for w in sys.argv[1:]:
    obs = load_saved(w)
    fp, Rf = pc.flange_pose(obs.state)
    allw = []
    for cam in ('front', 'wrist'):
        P = obs.points(cam)
        r, g, b = obs.colors(cam)
        m = pc.grey_mask(r, g, b) & np.isfinite(P[..., 2]) & (np.linalg.norm(P - fp, axis=-1) < 0.30) & (P[..., 2] > nail_z + 0.012)
        pts = P[m]
        Lf = (pts - fp) @ Rf
        pts = pts[Lf[:, 2] > 0.03]
        allw.append(pts)
        zmin = np.percentile(pts[:, 2], 1)
        for dz in (0.006, 0.015, 0.03):
            slab = pts[pts[:, 2] < zmin + dz]
            lo = np.percentile(slab[:, :2], 2, axis=0)
            hi = np.percentile(slab[:, :2], 98, axis=0)
            print(w, cam, 'zmin', round(zmin, 4), 'dz', dz, 'n', len(slab), 'x', lo[0].round(4), hi[0].round(4), 'y', lo[1].round(4), hi[1].round(4))
    W = np.concatenate(allw)
    zmin = np.percentile(W[:, 2], 1)
    for dz in (0.006, 0.015, 0.03):
        slab = W[W[:, 2] < zmin + dz]
        lo = np.percentile(slab[:, :2], 2, axis=0)
        hi = np.percentile(slab[:, :2], 98, axis=0)
        print(w, 'both', 'zmin', round(zmin, 4), 'dz', dz, 'x', lo[0].round(4), hi[0].round(4), 'y', lo[1].round(4), hi[1].round(4), 'centre', ((lo + hi) / 2).round(4))
