import sys, json
sys.path.insert(0, '/workspace')
import numpy as np
from geom import deproject, cam_mats
tag = sys.argv[1]
d = np.load(f'/workspace/data/{tag}.npz'); o = json.load(open(f'/workspace/data/{tag}_obs.json'))
K, T = cam_mats(o, 'robot0_eye_in_hand')
P = deproject(d['robot0_eye_in_hand_depth_m'], K, T)
c0 = np.array([-0.0076, -0.1908, 0.9884])
w = (np.hypot(P[..., 0] - c0[0], P[..., 1] - c0[1]) < 0.04) & (P[..., 2] > c0[2] - 0.03) & (P[..., 2] < c0[2] + 0.03)
q = P[w]; print('n', len(q), 'zmax', q[:, 2].max())
zt = np.percentile(q[:, 2], 99.5)
top = q[q[:, 2] > zt - 0.003]
print('top n', len(top), 'z', np.median(top[:, 2]).round(4))
ctr = top[:, :2].mean(0)
u, s, vt = np.linalg.svd(top[:, :2] - ctr)
maj = vt[0]
r = (top[:, :2] - ctr) @ vt.T
print('centre', ctr.round(4), 'sv', s.round(3), 'major yaw', np.degrees(np.arctan2(maj[1], maj[0])).round(2),
      'axis yaw', np.degrees(np.arctan2(vt[1, 1], vt[1, 0])).round(2), 'ext', (r.max(0) - r.min(0)).round(4))
vs, us = np.nonzero(w & (P[..., 2] > zt - 0.003)); print('rows', vs.min(), vs.max(), 'cols', us.min(), us.max())
