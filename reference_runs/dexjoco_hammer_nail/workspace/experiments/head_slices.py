"""Grey (head) points from the wrist cloud in the flange frame; slices along flange z."""
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
grey = (abs(r - g) < 14) & (abs(g - b) < 16) & (r > 50) & (r < 235)
near = np.linalg.norm(P - fp, axis=-1) < 0.30
m = grey & near
L = Rf.inv().apply(P[m] - fp)
W = P[m]
print('n', len(L), 'flange-frame min', L.min(0).round(4), 'max', L.max(0).round(4))
for z0 in np.arange(0.0, L[:, 2].max() + 0.01, 0.02):
    s = (L[:, 2] >= z0) & (L[:, 2] < z0 + 0.02)
    if s.sum() > 20:
        print(f'  zf {z0:.2f}: n={s.sum():5d} xf [{np.percentile(L[s,0],2):+.4f},{np.percentile(L[s,0],98):+.4f}] yf [{np.percentile(L[s,1],2):+.4f},{np.percentile(L[s,1],98):+.4f}]  world z [{W[s,2].min():.4f},{W[s,2].max():.4f}]')
lo = W[:, 2] < np.percentile(W[:, 2], 3)
print('lowest 3% world pts mean', W[lo].mean(0).round(4), 'flange', L[lo].mean(0).round(4))
