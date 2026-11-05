import sys
sys.path.insert(0, 'experiments')
from saved import *

tag = sys.argv[1]
cam = sys.argv[2] if len(sys.argv) > 2 else 'wrist'
rgb, dep, P, meta = cloud(tag, cam)
r, g, b = [rgb[..., i].astype(int) for i in range(3)]
z = P[..., 2]
table_z = np.percentile(z[(r > 200) & (g > 200) & (b > 200)], 50)
ylim = float(sys.argv[3]) if len(sys.argv) > 3 else -0.3
wood = (r - b > 40) & (r > 80) & (z > table_z + 0.02) & (P[..., 1] < ylim)
pts = P[wood]
# iterative: fit line in xy, then keep points near the local top surface
c = pts.mean(0)
cov = np.cov((pts[:, :2] - c[:2]).T)
w, v = np.linalg.eigh(cov)
d = v[:, -1]
if d[0] > 0:
    d = -d
for it in range(3):
    a = (pts[:, :2] - c[:2]) @ d
    # local top: bin along axis
    keep = np.zeros(len(pts), bool)
    bins = np.arange(a.min(), a.max() + 0.01, 0.01)
    for lo in bins:
        s = (a >= lo) & (a < lo + 0.01)
        if s.sum() > 10:
            zt = np.percentile(pts[s, 2], 97)
            keep |= s & (pts[:, 2] > zt - 0.004)
    tp = pts[keep]
    c = tp.mean(0)
    cov = np.cov((tp[:, :2] - c[:2]).T)
    w, v = np.linalg.eigh(cov)
    d = v[:, -1]
    if d[0] > 0:
        d = -d
a = (tp[:, :2] - c[:2]) @ d
perp = np.array([-d[1], d[0]])
lat = (tp[:, :2] - c[:2]) @ perp
A = np.stack([a, np.ones_like(a)], 1)
slope, z0 = np.linalg.lstsq(A, tp[:, 2], rcond=None)[0]
print(f'table_z {table_z:.4f}  top-face pts {len(tp)}')
print('axis xy dir', d.round(4), 'yaw deg', np.degrees(np.arctan2(d[1], d[0])).round(2), 'top centroid', c.round(4))
print(f'top-face z = {z0:.4f} + {slope:.4f}*a ; incline deg {np.degrees(np.arctan(slope)):.1f} ; a range [{a.min():.3f},{a.max():.3f}]')
print('top-face lateral width pct', np.percentile(lat, [2, 50, 98]).round(4))
