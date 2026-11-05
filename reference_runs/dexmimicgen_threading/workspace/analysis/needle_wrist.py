import sys, json
sys.path.insert(0, '/workspace')
import numpy as np
from geom import deproject, cam_mats
tag = sys.argv[1]; cam = sys.argv[2] if len(sys.argv) > 2 else 'robot1_eye_in_hand'
d = np.load(f'/workspace/data/{tag}.npz'); o = json.load(open(f'/workspace/data/{tag}_obs.json'))
K, T = cam_mats(o, cam)
P = deproject(d[cam + '_depth_m'], K, T)
c0 = np.array([-0.1363, 0.1675]); zt = 0.8005
w = (np.hypot(P[..., 0] - c0[0], P[..., 1] - c0[1]) < 0.05) & (P[..., 2] > zt + 0.032) & (P[..., 2] < zt + 0.05)
top = P[w]; print('top n', len(top), 'z', np.median(top[:, 2]).round(4))
best = None
for a in np.radians(np.arange(0, 90, 0.25)):
    R2 = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    r = top[:, :2] @ R2.T
    lo, hi = r.min(0), r.max(0)
    area = np.prod(hi - lo)
    if best is None or area < best[0]: best = (area, a, lo, hi, R2)
area, a, lo, hi, R2 = best
ctr = R2.T @ ((lo + hi) / 2)
print('cube rect yaw', np.degrees(a).round(2), 'ext', (hi - lo).round(4), 'centre', ctr.round(4), 'centroid', top[:, :2].mean(0).round(4))
sh = (np.hypot(P[..., 0] - ctr[0], P[..., 1] - ctr[1]) > 0.03) & (np.hypot(P[..., 0] - ctr[0], P[..., 1] - ctr[1]) < 0.17) & (P[..., 2] > zt + 0.012) & (P[..., 2] < zt + 0.03)
s = P[sh]; print('shaft n', len(s))
if len(s) > 10:
    v = s[:, :2].mean(0) - ctr; print('shaft yaw', np.degrees(np.arctan2(v[1], v[0])).round(2))
