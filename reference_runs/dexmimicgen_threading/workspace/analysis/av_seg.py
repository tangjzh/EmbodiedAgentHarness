import sys, json
sys.path.insert(0, '/workspace')
import numpy as np
from geom import deproject, cam_mats
tag = sys.argv[1] if len(sys.argv) > 1 else 'grab'
d = np.load(f'/workspace/data/{tag}.npz'); o = json.load(open(f'/workspace/data/{tag}_obs.json'))
K, T = cam_mats(o, 'agentview')
P = deproject(d['agentview_depth_m'], K, T); rgb = d['agentview_rgb'].astype(float)
z = P[..., 2]
# table height: mode of z over region x>-0.3
m = (P[..., 0] > -0.3) & (z > 0.7) & (z < 0.9)
h, e = np.histogram(z[m], bins=np.arange(0.7, 0.9, 0.001)); zt = e[h.argmax()] + 0.0005
print('table z', zt, 'count', h.max())
above = (z > zt + 0.004) & (z < zt + 0.25) & (P[..., 0] > -0.4) & (np.abs(P[..., 1]) < 0.5)
br = rgb.mean(-1)
print('above px', above.sum())
# colour stats for above points
for name, mm in [('dark', above & (br < 90)), ('light', above & (br >= 90))]:
    p = P[mm]
    if len(p): print(name, len(p), 'x', p[:, 0].min().round(3), p[:, 0].max().round(3), 'y', p[:, 1].min().round(3), p[:, 1].max().round(3), 'z', p[:, 2].min().round(3), p[:, 2].max().round(3), 'rgb', rgb[mm].mean(0).round(0))
