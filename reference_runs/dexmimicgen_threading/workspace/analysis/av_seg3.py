import sys, json
sys.path.insert(0, '/workspace')
import numpy as np
from scipy import ndimage
from geom import deproject, cam_mats
tag = sys.argv[1] if len(sys.argv) > 1 else 'grab'
d = np.load(f'/workspace/data/{tag}.npz'); o = json.load(open(f'/workspace/data/{tag}_obs.json'))
K, T = cam_mats(o, 'agentview')
P = deproject(d['agentview_depth_m'], K, T); rgb = d['agentview_rgb'].astype(float)
z = P[..., 2]
m = (P[..., 0] > -0.3) & (z > 0.7) & (z < 0.9)
h, e = np.histogram(z[m], bins=np.arange(0.7, 0.9, 0.001)); zt = e[h.argmax()] + 0.0005
cand = (z > zt + 0.004) & (z < zt + 0.215) & (P[..., 0] > -0.35) & (np.abs(P[..., 1]) < 0.45)
for i in range(2):
    ee = np.array(o['robot'][f'robot{i}_eef_pos'])
    cand &= ~((np.hypot(P[..., 0] - ee[0], P[..., 1] - ee[1]) < 0.07) & (z > ee[2] - 0.04))
p = P[cand]; c = rgb[cand]
# grid label in xy
res = 0.005
ij = np.floor((p[:, :2] - [-0.4, -0.5]) / res).astype(int)
G = np.zeros((int(0.7 / res) + 2, int(1.0 / res) + 2), bool); G[ij[:, 0], ij[:, 1]] = True
G = ndimage.binary_closing(G, iterations=2) | G
lab, n = ndimage.label(G)
pl = lab[ij[:, 0], ij[:, 1]]
comps = []
for k in range(1, n + 1):
    s = pl == k
    if s.sum() < 15: continue
    q = p[s]
    comps.append(dict(k=k, n=int(s.sum()), zmax=q[:, 2].max() - zt, xy=q[:, :2].mean(0), br=c[s].mean()))
for cc in comps: print(cc)
tri = max(comps, key=lambda cc: cc['zmax'])
nd = [cc for cc in comps if cc['zmax'] < 0.07 and cc['br'] < 100]
big = max(nd, key=lambda cc: cc['n'])
nd = [cc for cc in nd if np.linalg.norm(cc['xy'] - big['xy']) < 0.2]
sn = np.isin(pl, [cc['k'] for cc in nd]); q = p[sn]; hq = q[:, 2] - zt
top = q[hq > 0.035]
cc = top[:, :2].mean(0)
print('cube top n', len(top), 'centre', cc.round(4), 'z', np.median(top[:, 2]).round(4))
sh = q[(hq < 0.03) & (np.linalg.norm(q[:, :2] - cc, axis=1) > 0.035)]
v = sh[:, :2].mean(0) - cc
# PCA of shaft points for direction
u, s_, vt = np.linalg.svd(sh[:, :2] - sh[:, :2].mean(0)); dirn = vt[0] * np.sign(vt[0] @ v)
print('shaft n', len(sh), 'yaw(mean)', np.degrees(np.arctan2(v[1], v[0])).round(1), 'yaw(pca)', np.degrees(np.arctan2(dirn[1], dirn[0])).round(1),
      'far end', (sh[:, :2] @ dirn).max() - cc @ dirn)
# tripod
st = pl == tri['k']; q = p[st]; hq = q[:, 2] - zt
ring = q[hq > 0.165]
print('ring n', len(ring), 'centroid', ring.mean(0).round(4), 'zmax', q[:, 2].max().round(4))
u, s_, vt = np.linalg.svd(ring[:, :2] - ring[:, :2].mean(0)); print('ring pca sv', s_.round(3), 'major yaw', np.degrees(np.arctan2(vt[0, 1], vt[0, 0])).round(1))
plate = q[(hq > 0.075) & (hq < 0.095)]
print('plate n', len(plate), 'z', np.median(plate[:, 2]).round(4))
# plate orientation: min-area-rect via rotating calipers over angles
best = None
for a in np.radians(np.arange(0, 90, 0.5)):
    R2 = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    r = plate[:, :2] @ R2.T
    ext = np.percentile(r, 99, 0) - np.percentile(r, 1, 0)
    area = ext[0] * ext[1]
    if best is None or area < best[0]: best = (area, np.degrees(a), ext)
print('plate rect yaw', best[1], 'ext', best[2].round(3))
