import sys
sys.path.insert(0, 'experiments')
from saved import *

tag = sys.argv[1]
cam = sys.argv[2] if len(sys.argv) > 2 else 'wrist'
rgb, dep, P, meta = cloud(tag, cam)
r, g, b = [rgb[..., i].astype(int) for i in range(3)]
z = P[..., 2]
table_z = np.percentile(z[(r > 200) & (g > 200) & (b > 200)], 50)
print('table z (white median)', round(table_z, 4))
wood = (r - b > 40) & (r > 80) & (z > table_z + 0.02)
grey = (abs(r - g) < 14) & (abs(g - b) < 16) & (r > 50) & (r < 235) & (z > table_z + 0.005)
for name, m in (('wood', wood), ('grey', grey)):
    pts = P[m]
    if len(pts) == 0:
        print(name, 'none'); continue
    print(name, m.sum(), 'min', pts.min(0).round(4), 'max', pts.max(0).round(4), 'mean', pts.mean(0).round(4))
pts = P[wood]
c = pts.mean(0)
u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
d = vt[0] if vt[0][0] < 0 else -vt[0]
print('wood axis dir', d.round(4), 'sv', s.round(3))
# cross-section: project onto plane perpendicular to d (horizontal perp and z)
perp = np.cross([0, 0, 1], d); perp /= np.linalg.norm(perp)
a = (pts - c) @ d
lat = (pts - c) @ perp
print('axis extent', a.min().round(4), a.max().round(4))
print('lateral extent', np.percentile(lat, [1, 50, 99]).round(4), 'z extent', np.percentile(pts[:, 2], [1, 50, 99]).round(4))
# top surface z along axis
for t in np.linspace(a.min(), a.max(), 8):
    sel = abs(a - t) < 0.01
    if sel.sum() > 5:
        print(f'  t={t:+.3f} n={sel.sum():5d} ztop={np.percentile(pts[sel,2],98):.4f} lat=[{np.percentile(lat[sel],2):+.4f},{np.percentile(lat[sel],98):+.4f}] center xy={(c + d*t)[:2].round(4)}')
gp = P[grey]
# head: grey cluster near wood end with min a
hp = gp[np.linalg.norm(gp[:, :2] - (c + d * a.min())[:2], axis=1) < 0.08]
if len(hp):
    print('head pts', len(hp), 'min', hp.min(0).round(4), 'max', hp.max(0).round(4), 'center xy', ((hp.min(0) + hp.max(0)) / 2).round(4))
