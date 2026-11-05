"""Hand geometry in the flange frame from a saved call's point cloud (front + wrist)."""
import sys
sys.path.insert(0, 'experiments')
from saved import *
from scipy.spatial.transform import Rotation as R

tag = sys.argv[1]
cam = sys.argv[2] if len(sys.argv) > 2 else 'front'
rgb, dep, P, meta = cloud(tag, cam)
st = meta['state']
fp = np.array(st[0:3])
q = st[3:7]
Rf = R.from_quat([q[1], q[2], q[3], q[0]])
pts = P.reshape(-1, 3)
cols = rgb.reshape(-1, 3).astype(int)
loc = Rf.inv().apply(pts - fp)
near = (np.linalg.norm(loc, axis=1) < 0.30) & (loc[:, 2] > -0.06)
dark = cols.max(1) < 90  # black fingers / palm
sel = near & dark
L = loc[sel]
print('hand pts', len(L))
if len(L):
    print('flange-frame extent min', L.min(0).round(4), 'max', L.max(0).round(4))
    # along finger direction (x) bins: lowest (max z in flange frame = farthest from flange plane downward)
    for x0 in np.arange(-0.04, 0.24, 0.02):
        s = (L[:, 0] >= x0) & (L[:, 0] < x0 + 0.02)
        if s.sum() > 3:
            print(f'  x {x0:+.2f}: n={s.sum():4d} z(depth below flange) min {L[s,2].min():.4f} max {L[s,2].max():.4f}  y [{L[s,1].min():+.4f},{L[s,1].max():+.4f}]')
