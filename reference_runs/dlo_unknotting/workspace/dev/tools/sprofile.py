"""Height / lateral profile of the rope along y (relative to P) for capture dirs: per 1 cm y-bin, rope point max z,
x-extent (min/max x - P_x) and #points. usage: python3 tools/sprofile.py capdir..."""
import json
import sys

import numpy as np

sys.path.insert(0, "/workspace")
import rope_vision as rv  # noqa: E402

for d in sys.argv[1:]:
    d = d.rstrip("/") + "/"
    obs = json.load(open(d + "obs.json"))
    cams = {}
    for n in rv.CAMS:
        cams[n + "_rgb"] = np.load(d + n + "_rgb.npy")
        cams[n + "_depth_m"] = np.load(d + n + "_depth_m.npy")
    P = np.array(obs["robot"]["right"]["eef_position_world"])
    pts, _ = rv.rope_points(obs, cams)
    rel = pts - P
    ok = (np.abs(rel[:, 0]) < 0.25)
    rel = rel[ok]
    print(d)
    rows = []
    for y in np.arange(-0.70, 0.30, 0.02):
        m = (rel[:, 1] >= y) & (rel[:, 1] < y + 0.02)
        if m.sum() < 5:
            rows.append("%+.2f: --" % y)
            continue
        z = pts[ok][m, 2]
        rows.append("%+.2f: z%4.1f x[%+5.1f,%+5.1f]" % (y, np.percentile(z, 98) * 100, rel[m, 0].min() * 100, rel[m, 0].max() * 100))
    for i in range(0, len(rows), 5):
        print("   " + " | ".join(rows[i:i + 5]))
