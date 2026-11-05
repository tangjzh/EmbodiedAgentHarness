# Code block 0
# Step 1: perceive the scene and estimate the red cube's 3D position in the world frame.
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
T_wc = cam["pose_mat"]
print("rgb", rgb.shape, "depth", depth.shape, depth.dtype, "depth range", np.nanmin(depth), np.nanmax(depth))
print("K", K)
print("pose_mat", T_wc)

res = segment_sam3_text_prompt(rgb, "red cube")
print("num masks", len(res), [r.get("score") for r in res])
best = max(res, key=lambda d: d["score"])
mask = best["mask"]
print("box", best.get("box"), "mask pixels", mask.sum())

# Back-project the masked depth pixels into camera coordinates (OpenCV convention).
ys, xs = np.nonzero(mask)
d = depth[ys, xs].astype(np.float64)
valid = np.isfinite(d) & (d > 0)
ys, xs, d = ys[valid], xs[valid], d[valid]
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
pts_c = np.stack([(xs - cx) * d / fx, (ys - cy) * d / fy, d], axis=1)

# Transform the points into the world frame.
pts_w = (T_wc[:3, :3] @ pts_c.T).T + T_wc[:3, 3]
print("cube pts world mean", pts_w.mean(0), "min", pts_w.min(0), "max", pts_w.max(0))
obb = get_oriented_bounding_box_from_3d_points(pts_w)
print("obb", obb)

# Code block 1
import numpy as np

# Top face of the cube: points close to the max z
zmax = pts_w[:, 2].max()
top = pts_w[pts_w[:, 2] > zmax - 0.01]
print("top pts", len(top), "zmax", zmax)

# Estimate yaw by minimizing the xy bounding-box area of the top-face points
best_area, best_th = 1e9, 0.0
for th in np.linspace(-np.pi/4, np.pi/4, 91):
    c, s = np.cos(th), np.sin(th)
    R2 = np.array([[c, s], [-s, c]])
    p = top[:, :2] @ R2.T
    area = np.ptp(p[:, 0]) * np.ptp(p[:, 1])
    if area < best_area:
        best_area, best_th = area, th
c, s = np.cos(best_th), np.sin(best_th)
R2 = np.array([[c, s], [-s, c]])
p = top[:, :2] @ R2.T
ctr_rot = np.array([(p[:, 0].min() + p[:, 0].max()) / 2, (p[:, 1].min() + p[:, 1].max()) / 2])
ctr_xy = R2.T @ ctr_rot
print("yaw", np.degrees(best_th), "center xy", ctr_xy, "sizes", np.ptp(p[:, 0]), np.ptp(p[:, 1]))

# Grasp the cube about 2 cm below its top face
grasp_z = zmax - 0.02
FINGER_OFFSET = 0.1034  # distance from panda_hand to fingertip center
quat = np.array([0.0, np.cos(best_th / 2), np.sin(best_th / 2), 0.0])  # gripper pointing down, rotated by yaw

def go(pos):
    j = solve_ik(np.asarray(pos, dtype=np.float64), quat)
    move_to_joints(j)

open_gripper()
pre = [ctr_xy[0], ctr_xy[1], grasp_z + FINGER_OFFSET + 0.15]
go(pre)
go([ctr_xy[0], ctr_xy[1], grasp_z + FINGER_OFFSET + 0.05])
go([ctr_xy[0], ctr_xy[1], grasp_z + FINGER_OFFSET])
close_gripper()
go([ctr_xy[0], ctr_xy[1], grasp_z + FINGER_OFFSET + 0.2])

# Verify the cube has been lifted
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
res = segment_sam3_text_prompt(rgb, "red cube")
if len(res):
    b = max(res, key=lambda d: d["score"])
    ys, xs = np.nonzero(b["mask"])
    d = depth[ys, xs].astype(np.float64)
    v = np.isfinite(d) & (d > 0)
    ys, xs, d = ys[v], xs[v], d[v]
    pc = np.stack([(xs - cx) * d / fx, (ys - cy) * d / fy, d], axis=1)
    pw = (cam["pose_mat"][:3, :3] @ pc.T).T + cam["pose_mat"][:3, 3]
    print("score", b["score"], "cube pts world mean after lift", pw.mean(0))
else:
    print("cube not found")