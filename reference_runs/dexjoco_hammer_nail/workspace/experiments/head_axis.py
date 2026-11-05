"""Estimate hammer head axis & striking face in the flange frame from a saved call (front camera)."""
import sys
sys.path.insert(0, 'experiments')
from saved import *
from scipy.spatial.transform import Rotation as R

tag = sys.argv[1]
cam = sys.argv[2] if len(sys.argv) > 2 else 'front'
rgb, dep, P, meta = cloud(tag, cam)
st = meta['state']
fp = np.array(st[0:3]); q = st[3:7]
Rf = R.from_quat([q[1], q[2], q[3], q[0]])
r, g, b = [rgb[..., i].astype(int) for i in range(3)]
grey = (abs(r - g) < 14) & (abs(g - b) < 16) & (r > 60) & (r < 235)
near = np.linalg.norm(P - fp, axis=-1) < 0.35
m = grey & near & (P[..., 2] > 0.97)
pts = P[m]
L = Rf.inv().apply(pts - fp)
print('n', len(pts))
c = pts.mean(0)
u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
ax = vt[0] if vt[0][2] < 0 else -vt[0]  # pointing down
proj = (pts - c) @ ax
print('head centroid world', c.round(4), 'axis (down) world', ax.round(4), 'tilt from vertical deg', np.degrees(np.arccos(-ax[2])).round(1))
print('extent along axis', proj.min().round(4), proj.max().round(4), 'sv', s.round(3))
bottom = c + ax * proj.max()
top = c + ax * proj.min()
print('bottom (striking end) world', bottom.round(4), ' top world', top.round(4))
print('bottom in flange frame', Rf.inv().apply(bottom - fp).round(4), ' axis in flange frame', Rf.inv().apply(ax).round(4))
