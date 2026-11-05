"""Wide top-down height maps of the whole rope for several capture dirs, stacked vertically.
usage: python3 tools/widetop.py out.png capdir...   (frame relative to P: x in [-0.2,0.2], y in [-0.8,0.35])"""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "/workspace")
import rope_vision as rv  # noqa: E402

out, dirs = sys.argv[1], sys.argv[2:]
RES = 0.0025
tiles = []
for d in dirs:
    d = d.rstrip("/") + "/"
    obs = json.load(open(d + "obs.json"))
    cams = {}
    for n in rv.CAMS:
        cams[n + "_rgb"] = np.load(d + n + "_rgb.npy")
        cams[n + "_depth_m"] = np.load(d + n + "_depth_m.npy")
    P = np.array(obs["robot"]["right"]["eef_position_world"]); Q = np.array(obs["robot"]["left"]["eef_position_world"])
    pts, _ = rv.rope_points(obs, cams)
    x0, x1, y0, y1 = P[0] - 0.2, P[0] + 0.2, P[1] - 0.8, P[1] + 0.35
    zb = rv.height_map(pts, x0, x1, y0, y1, RES)
    img = rv.render_height(zb, 0.06, [(P[0], P[1]), (Q[0], Q[1])], x1, y0, RES)
    # 5 cm grid ticks along the P line
    for yy in np.arange(-0.8, 0.35, 0.05):
        c = int((yy + 0.8) / RES)
        cv2.line(img, (c, 0), (c, 4), (90, 90, 90), 1)
    cv2.line(img, (0, int(0.2 / RES)), (img.shape[1] - 1, int(0.2 / RES)), (60, 60, 60), 1)
    cv2.putText(img, "%s P=(%.3f,%.3f,%.3f) Q=(%.3f,%.3f,%.3f) |PQ|=%.3f" % (d.split("out/")[-1], *P, *Q, np.linalg.norm(P - Q)),
                (4, img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    tiles.append(img)
cv2.imwrite(out, np.vstack(tiles))
print("wrote", out)
