"""Locate the nail head from a front image: grey thin object above the board top."""
import sys
sys.path.insert(0, 'experiments')
from saved import *

tag = sys.argv[1]
rgb, dep, P, meta = cloud(tag, 'front')
r, g, b = [rgb[..., i].astype(int) for i in range(3)]
z = P[..., 2]
grey = (abs(r - g) < 16) & (abs(g - b) < 18) & (r > 60) & (r < 240)
# board region: brown pixels with world z high; board top z = median z of brown top surface
brown = (r - b > 25) & (r < 200) & (g < 170)
bz = z[brown & (P[..., 0] < 0.04) & (P[..., 0] > -0.25) & (abs(P[..., 1]) < 0.25)]
board_top = np.percentile(bz, 90)
print('board top z ~', round(board_top, 4))
cand = grey & (z > board_top + 0.005) & (z < board_top + 0.10) & (abs(P[..., 1]) < 0.3) & (P[..., 0] > -0.3) & (P[..., 0] < 0.1)
bx = P[brown & (abs(z - board_top) < 0.004)]
print('board top extents x', np.percentile(bx[:, 0], [1, 99]).round(3), 'y', np.percentile(bx[:, 1], [1, 99]).round(3))
cand &= (P[..., 0] > np.percentile(bx[:, 0], 1)) & (P[..., 0] < np.percentile(bx[:, 0], 99) + 0.01)
cand &= (P[..., 1] > np.percentile(bx[:, 1], 1)) & (P[..., 1] < np.percentile(bx[:, 1], 99))
pts = P[cand]
print('nail cand', len(pts))
if len(pts):
    top = pts[pts[:, 2] > pts[:, 2].max() - 0.006]
    print('max z', pts[:, 2].max().round(4), 'top pts', len(top), 'top mean', top.mean(0).round(4))
    print('all cand mean', pts.mean(0).round(4), 'x range', pts[:, 0].min().round(4), pts[:, 0].max().round(4), 'y range', pts[:, 1].min().round(4), pts[:, 1].max().round(4))
    vs, us = np.nonzero(cand)
    print('pixel bbox u', us.min(), us.max(), 'v', vs.min(), vs.max())
