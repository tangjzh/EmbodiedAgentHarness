import sys
sys.path.insert(0, 'experiments')
from geom import *

call = sys.argv[1] if len(sys.argv) > 1 else '000001'
cd, resp = load_call(latest_run() / 'calls' / call)
meta = find_obs_meta(resp)
rgb = np.array(Image.open(cd / 'image_01.png')).astype(np.int32)
dep = np.array(Image.open(cd / 'image_03.png')).astype(np.float64) * 0.001
P = cam_points(dep, meta['camera_intrinsics']['front']['matrix'], meta['camera_poses']['front'])
r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
orange = (r > 150) & (g > 90) & (b < 80) & (r - b > 90)
pts = P[orange]
c = pts.mean(0)
u, s, vt = np.linalg.svd(pts - c)
d = vt[0]
if d[0] > 0:
    d = -d
proj = (pts - c) @ d
print('handle centroid', c.round(4), 'dir', d.round(4), 'extent along axis', proj.min().round(4), proj.max().round(4))
print('yaw deg', np.degrees(np.arctan2(d[1], d[0])).round(2))
vs, us = np.nonzero(orange)
print('pixel bbox u', us.min(), us.max(), 'v', vs.min(), vs.max())
# metal grey head region: grey pixels whose 3D point is within 0.12 m of the handle end nearest the board
grey = (abs(r - g) < 14) & (abs(g - b) < 16) & (r > 50) & (r < 235)
near = np.linalg.norm(P - (c + d * proj.min()), axis=-1) < 0.12
hm = grey & near & (P[..., 2] > 0.96)
hp = P[hm]
print('head px', hm.sum(), 'mean', hp.mean(0).round(4), 'min', hp.min(0).round(4), 'max', hp.max(0).round(4))
vs, us = np.nonzero(hm)
print('head pixel bbox u', us.min(), us.max(), 'v', vs.min(), vs.max())
