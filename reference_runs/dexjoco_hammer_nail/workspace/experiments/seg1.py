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
white = (r > 215) & (g > 215) & (b > 215)
z = P[..., 2]
print('white px', white.sum(), 'z pct of white:', np.percentile(z[white], [5, 25, 50, 75, 95]).round(4))
orange = (r > 150) & (g > 90) & (b < 80) & (r - b > 90)
print('orange px', orange.sum())
if orange.sum():
    pts = P[orange]
    print('orange pts mean', pts.mean(0).round(4), 'min', pts.min(0).round(4), 'max', pts.max(0).round(4))
grey = (abs(r - g) < 12) & (abs(g - b) < 14) & (r > 60) & (r < 200)
boxes = [('hammerhead', (40, 150, 220, 330)), ('nail', (300, 360, 240, 300))]
for name, (u0, u1, v0, v1) in boxes:
    m = np.zeros_like(grey)
    m[v0:v1, u0:u1] = grey[v0:v1, u0:u1]
    pts = P[m]
    print(name, m.sum(), 'mean', pts.mean(0).round(4), 'min', pts.min(0).round(4), 'max', pts.max(0).round(4))
