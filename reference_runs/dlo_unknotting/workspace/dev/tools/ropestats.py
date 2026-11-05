"""Offline: print rope point statistics in slabs along y relative to P (to locate knot/tail height)."""
import json
import sys

import numpy as np

sys.path.insert(0, '/workspace')
import rope_vision as rv

d = sys.argv[1].rstrip('/') + '/'
obs = json.load(open(d + 'obs.json'))
cams = {}
for n in rv.CAMS:
    cams[n + '_rgb'] = np.load(d + n + '_rgb.npy')
    cams[n + '_depth_m'] = np.load(d + n + '_depth_m.npy')
P = np.array(obs['robot']['right']['eef_position_world'])
Q = np.array(obs['robot']['left']['eef_position_world'])
pts, ci = rv.rope_points(obs, cams)
print('P', np.round(P, 3), 'Q', np.round(Q, 3), 'npts', len(pts))
rel = pts - P
for y0 in np.arange(-0.5, 0.32, 0.02):
    m = (rel[:, 1] >= y0) & (rel[:, 1] < y0 + 0.02)
    if m.sum() < 5:
        continue
    z = pts[m, 2]; x = rel[m, 0]
    print('dy[%+.2f,%+.2f) n=%5d  z min %.3f med %.3f max %.3f | dx min %+.3f max %+.3f | cams %s' % (
        y0, y0 + 0.02, m.sum(), z.min(), np.median(z), z.max(), x.min(), x.max(), np.bincount(ci[m], minlength=4).tolist()))
