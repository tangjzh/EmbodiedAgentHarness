import sys, json
sys.path.insert(0, '/workspace')
import numpy as np
from geom import deproject, cam_mats
tag = sys.argv[1] if len(sys.argv) > 1 else 'grab'
d = np.load(f'/workspace/data/{tag}.npz'); o = json.load(open(f'/workspace/data/{tag}_obs.json'))
K, T = cam_mats(o, 'agentview')
P = deproject(d['agentview_depth_m'], K, T); rgb = d['agentview_rgb'].astype(float)
z = P[..., 2]; zt = 0.8005
cand = (z > zt + 0.004) & (z < 1.02) & (P[..., 0] > -0.35) & (np.abs(P[..., 1]) < 0.45)
for i in range(2):
    e = np.array(o['robot'][f'robot{i}_eef_pos'])
    near = (np.hypot(P[..., 0] - e[0], P[..., 1] - e[1]) < 0.07) & (z > e[2] - 0.04)
    cand &= ~near
br = rgb.mean(-1)
sat = rgb.max(-1) - rgb.min(-1)
dark = cand & (br < 100); light = cand & (br >= 100)
for name, mm in [('dark', dark), ('light', light)]:
    p = P[mm]
    print(name, len(p), 'x', np.percentile(p[:, 0], [0, 50, 100]).round(3), 'y', np.percentile(p[:, 1], [0, 50, 100]).round(3), 'z', np.percentile(p[:, 2], [0, 50, 100]).round(3), 'rgb', rgb[mm].mean(0).round(0))
vs, us = np.nonzero(dark); print('dark rows', vs.min(), vs.max(), 'cols', us.min(), us.max())
vs, us = np.nonzero(light); print('light rows', vs.min(), vs.max(), 'cols', us.min(), us.max())
from PIL import Image
im = d['agentview_rgb'].copy(); im[dark] = [255, 0, 0]; im[light] = [0, 255, 0]
Image.fromarray(im).resize((640, 480), Image.NEAREST).save('/workspace/analysis/av_seg.png')
