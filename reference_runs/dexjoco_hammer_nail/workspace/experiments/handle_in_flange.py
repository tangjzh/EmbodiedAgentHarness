"""Express handle (wood) and head (grey) points in the current flange frame."""
import sys
sys.path.insert(0, 'experiments')
from saved import *
from scipy.spatial.transform import Rotation as R

tag = sys.argv[1]
cam = sys.argv[2] if len(sys.argv) > 2 else 'wrist'
rgb, dep, P, meta = cloud(tag, cam)
st = meta['state']
fp = np.array(st[0:3]); q = st[3:7]
Rf = R.from_quat([q[1], q[2], q[3], q[0]])
r, g, b = [rgb[..., i].astype(int) for i in range(3)]
z = P[..., 2]
wood = (r - b > 40) & (r > 80) & (z > 0.975) & (P[..., 1] < -0.25)
grey = (abs(r - g) < 14) & (abs(g - b) < 16) & (r > 50) & (r < 235) & (z > 0.965) & (P[..., 1] < -0.25) & (P[..., 1] > -0.7)
for name, m in (('wood', wood), ('grey', grey)):
    L = Rf.inv().apply(P[m] - fp)
    if len(L) == 0:
        print(name, 'none'); continue
    print(name, len(L), 'flange-frame min', L.min(0).round(4), 'max', L.max(0).round(4))
    if name == 'wood':
        for y0 in np.arange(np.floor(L[:, 1].min() * 50) / 50, L[:, 1].max(), 0.02):
            s = (L[:, 1] >= y0) & (L[:, 1] < y0 + 0.02)
            if s.sum() > 20:
                print(f'   yf {y0:+.2f}: x [{np.percentile(L[s,0],2):+.4f},{np.percentile(L[s,0],98):+.4f}]  z(top=min) {np.percentile(L[s,2],2):.4f}  zmax {np.percentile(L[s,2],98):.4f}')
