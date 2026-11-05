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

# Code block 2
import numpy as np

def perceive(prompt):
    obs = get_observation()
    cam = obs["robot0_robotview"]
    rgb = cam["images"]["rgb"]; depth = cam["images"]["depth"]
    K = cam["intrinsics"]; T = cam["pose_mat"]
    res = segment_sam3_text_prompt(rgb, prompt)
    if not res:
        return None, None
    b = max(res, key=lambda d: d["score"])
    ys, xs = np.nonzero(b["mask"])
    d = depth[ys, xs].astype(np.float64)
    v = np.isfinite(d) & (d > 0)
    ys, xs, d = ys[v], xs[v], d[v]
    pc = np.stack([(xs - K[0,2]) * d / K[0,0], (ys - K[1,2]) * d / K[1,1], d], axis=1)
    pw = (T[:3,:3] @ pc.T).T + T[:3,3]
    return pw, b["score"]

cube, sc = perceive("red cube")
zmax = cube[:,2].max()
top = cube[cube[:,2] > zmax - 0.01]
ctr = (top[:,:2].min(0) + top[:,:2].max(0)) / 2
print("cube center", ctr, "zmax", zmax, "zmin", cube[:,2].min())

th = np.radians(-21.0)
quat = np.array([0.0, np.cos(th/2), np.sin(th/2), 0.0])
FO = 0.1034
def go(pos):
    j = solve_ik(np.asarray(pos, dtype=np.float64), quat)
    print("target", np.round(pos,3), "joints", np.round(j,3))
    move_to_joints(j)

open_gripper()
# Hover 10cm above cube top (fingertips), then check where the gripper actually is
hover = [ctr[0], ctr[1], zmax + FO + 0.10]
go(hover)
g, gs = perceive("robot gripper fingers")
if g is not None:
    print("gripper score", gs, "gripper pts mean", g.mean(0), "min", g.min(0), "max", g.max(0))
    # lowest gripper points ~ fingertips; estimate xy offset vs commanded
    tips = g[g[:,2] < g[:,2].min() + 0.02]
    print("fingertip est", tips.mean(0), "expected", [ctr[0], ctr[1], zmax + 0.10])
else:
    print("gripper not detected")

# Code block 3
import numpy as np

def cam_data():
    obs = get_observation()
    cam = obs["robot0_robotview"]
    return cam["images"]["rgb"], cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"]

def pix_to_world(u, v, depth, K, T, win=4, use_min=True):
    H, W = depth.shape
    u0, u1 = max(0, u - win), min(W, u + win + 1)
    v0, v1 = max(0, v - win), min(H, v + win + 1)
    patch = depth[v0:v1, u0:u1].astype(np.float64)
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size == 0:
        return None
    d = np.percentile(patch, 10) if use_min else np.median(patch)
    pc = np.array([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d])
    return T[:3, :3] @ pc + T[:3, 3]

def molmo_pt(rgb, prompt):
    r = point_prompt_molmo(rgb, prompt)
    vals = list(r.values()) if isinstance(r, dict) else [r]
    for p in vals:
        if p is not None and p[0] is not None:
            return int(p[0]), int(p[1])
    return None

th = np.radians(-21.0)
quat = np.array([0.0, np.cos(th / 2), np.sin(th / 2), 0.0])
FO = 0.1034

def go(pos):
    j = solve_ik(np.asarray(pos, dtype=np.float64), quat)
    move_to_joints(j)

# Cube estimate
rgb, depth, K, T = cam_data()
res = segment_sam3_text_prompt(rgb, "red cube")
b = max(res, key=lambda d: d["score"])
ys, xs = np.nonzero(b["mask"])
d = depth[ys, xs].astype(np.float64); v = np.isfinite(d) & (d > 0)
ys, xs, d = ys[v], xs[v], d[v]
pc = np.stack([(xs - K[0, 2]) * d / K[0, 0], (ys - K[1, 2]) * d / K[1, 1], d], 1)
cube = (T[:3, :3] @ pc.T).T + T[:3, 3]
zmax = cube[:, 2].max()
top = cube[cube[:, 2] > zmax - 0.01]
ctr = (top[:, :2].min(0) + top[:, :2].max(0)) / 2
print("cube ctr", ctr, "zmax", zmax)

# Calibration: command fingertip positions and observe where fingertips appear
open_gripper()
offsets = []
for dxy in [(0.0, 0.0), (-0.05, 0.05), (0.05, -0.05)]:
    tgt_tip = np.array([ctr[0] + dxy[0], ctr[1] + dxy[1], zmax + 0.10])
    go(tgt_tip + np.array([0, 0, FO]))
    rgb, depth, K, T = cam_data()
    p = molmo_pt(rgb, "the tip of the robot gripper finger")
    print("molmo pt", p)
    if p is None:
        continue
    w = pix_to_world(p[0], p[1], depth, K, T)
    print("commanded tip", np.round(tgt_tip, 3), "observed", None if w is None else np.round(w, 3))
    if w is not None:
        offsets.append(w - tgt_tip)

if offsets:
    off = np.median(np.array(offsets), axis=0)
else:
    off = np.zeros(3)
print("offsets", np.round(np.array(offsets), 3), "median", np.round(off, 3))

# Apply xy correction only if plausible
corr = off[:2] if np.linalg.norm(off[:2]) < 0.3 else np.zeros(2)
print("applied xy correction", corr)
gx, gy = ctr[0] - corr[0], ctr[1] - corr[1]
grasp_z = zmax - 0.02

go([gx, gy, grasp_z + FO + 0.10])
go([gx, gy, grasp_z + FO + 0.03])
go([gx, gy, grasp_z + FO])
close_gripper()
go([gx, gy, grasp_z + FO + 0.20])

rgb, depth, K, T = cam_data()
res = segment_sam3_text_prompt(rgb, "red cube")
if res:
    b = max(res, key=lambda d: d["score"])
    ys, xs = np.nonzero(b["mask"])
    d = depth[ys, xs].astype(np.float64); v = np.isfinite(d) & (d > 0)
    ys, xs, d = ys[v], xs[v], d[v]
    pc = np.stack([(xs - K[0, 2]) * d / K[0, 0], (ys - K[1, 2]) * d / K[1, 1], d], 1)
    pw = (T[:3, :3] @ pc.T).T + T[:3, 3]
    print("cube after lift: score", b["score"], "mean", pw.mean(0), "zmax", pw[:, 2].max())
else:
    print("cube not found")

# Code block 4
import numpy as np

def cam_data():
    obs = get_observation()
    cam = obs["robot0_robotview"]
    return cam["images"]["rgb"], cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"]

def pix_to_world(u, v, depth, K, T, win=3):
    H, W = depth.shape
    patch = depth[max(0,v-win):min(H,v+win+1), max(0,u-win):min(W,u+win+1)].astype(np.float64)
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size == 0:
        return None
    d = np.percentile(patch, 10)
    pc = np.array([(u - K[0,2]) * d / K[0,0], (v - K[1,2]) * d / K[1,1], d])
    return T[:3,:3] @ pc + T[:3,3]

def molmo_all(rgb, prompt):
    r = point_prompt_molmo(rgb, prompt)
    vals = list(r.values()) if isinstance(r, dict) else [r]
    out = []
    for p in vals:
        if isinstance(p, (list, tuple)) and len(p) == 2 and p[0] is not None:
            out.append((int(p[0]), int(p[1])))
        elif isinstance(p, (list, tuple)):
            for q in p:
                if isinstance(q, (list, tuple)) and len(q) == 2 and q[0] is not None:
                    out.append((int(q[0]), int(q[1])))
    return out

def observe_tip():
    rgb, depth, K, T = cam_data()
    pts = molmo_all(rgb, "the tips of the robot gripper fingers")
    ws = [pix_to_world(u, v, depth, K, T) for u, v in pts]
    ws = [w for w in ws if w is not None]
    if not ws:
        return None
    return np.mean(ws, axis=0)

def cube_est():
    rgb, depth, K, T = cam_data()
    res = segment_sam3_text_prompt(rgb, "red cube")
    b = max(res, key=lambda d: d["score"])
    ys, xs = np.nonzero(b["mask"])
    d = depth[ys, xs].astype(np.float64); v = np.isfinite(d) & (d > 0)
    ys, xs, d = ys[v], xs[v], d[v]
    pc = np.stack([(xs-K[0,2])*d/K[0,0], (ys-K[1,2])*d/K[1,1], d], 1)
    pw = (T[:3,:3] @ pc.T).T + T[:3,3]
    zmax = pw[:,2].max()
    top = pw[pw[:,2] > zmax - 0.01]
    return (top[:,:2].min(0) + top[:,:2].max(0)) / 2, zmax, pw[:,2].min()

th = np.radians(-21.0)
quat = np.array([0.0, np.cos(th/2), np.sin(th/2), 0.0])
FO = 0.1034
def go(tip):
    j = solve_ik(np.asarray(tip, dtype=np.float64) + np.array([0, 0, FO]), quat)
    move_to_joints(j)

ctr, zmax, zmin = cube_est()
print("cube", ctr, zmax, zmin)
target_tip = np.array([ctr[0], ctr[1], zmax - 0.02])

open_gripper()
cmd = np.array([ctr[0] + 0.008, ctr[1] + 0.045, zmax + 0.12])  # initial guess with previous xy correction
for it in range(10):
    go(cmd)
    tip = observe_tip()
    print(it, "cmd", np.round(cmd,3), "observed tip", None if tip is None else np.round(tip,3))
    if tip is None:
        break
    err = target_tip - tip
    # once above cube and xy aligned, descend; limit step sizes
    step = err.copy()
    if np.linalg.norm(err[:2]) > 0.01:
        step[2] = max(step[2], -0.03)  # descend slowly while correcting xy
    step = np.clip(step, -0.06, 0.06)
    if np.linalg.norm(err) < 0.012:
        break
    cmd = cmd + step
    if tip[2] < zmax - 0.01 and np.linalg.norm(err[:2]) < 0.012:
        break

print("final cmd", cmd)
close_gripper()
go(cmd + np.array([0, 0, 0.2]))

ctr2, zmax2, zmin2 = cube_est()
print("cube after lift ctr", ctr2, "zmax", zmax2, "zmin", zmin2, "before zmax", zmax)

# Code block 5
import numpy as np

def cam_data():
    obs = get_observation()
    cam = obs["robot0_robotview"]
    return cam["images"]["rgb"], cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"]

def mask_to_world(mask, depth, K, T):
    ys, xs = np.nonzero(mask)
    d = depth[ys, xs].astype(np.float64); v = np.isfinite(d) & (d > 0)
    ys, xs, d = ys[v], xs[v], d[v]
    pc = np.stack([(xs-K[0,2])*d/K[0,0], (ys-K[1,2])*d/K[1,1], d], 1)
    return (T[:3,:3] @ pc.T).T + T[:3,3]

def cube_est():
    rgb, depth, K, T = cam_data()
    res = segment_sam3_text_prompt(rgb, "red cube")
    if not res:
        return None
    b = max(res, key=lambda d: d["score"])
    pw = mask_to_world(b["mask"], depth, K, T)
    zmax = pw[:,2].max()
    top = pw[pw[:,2] > zmax - 0.01]
    # yaw estimate
    best_area, best_th = 1e9, 0.0
    for th in np.linspace(-np.pi/4, np.pi/4, 91):
        c, s = np.cos(th), np.sin(th)
        R2 = np.array([[c, s], [-s, c]])
        p = top[:, :2] @ R2.T
        a = np.ptp(p[:,0]) * np.ptp(p[:,1])
        if a < best_area:
            best_area, best_th = a, th
    c, s = np.cos(best_th), np.sin(best_th)
    R2 = np.array([[c, s], [-s, c]])
    p = top[:, :2] @ R2.T
    cr = np.array([(p[:,0].min()+p[:,0].max())/2, (p[:,1].min()+p[:,1].max())/2])
    ctr = R2.T @ cr
    return ctr, zmax, pw[:,2].min(), best_th

def gripper_tip_est(cube_ctr):
    rgb, depth, K, T = cam_data()
    best = None
    for prompt in ["robot gripper fingers", "robot gripper"]:
        res = segment_sam3_text_prompt(rgb, prompt)
        if res:
            best = max(res, key=lambda d: d["score"])
            break
    if best is None:
        return None
    pw = mask_to_world(best["mask"], depth, K, T)
    # remove outliers far away in xy
    pw = pw[np.linalg.norm(pw[:,:2] - np.median(pw[:,:2], 0), axis=1) < 0.15]
    zmin = np.percentile(pw[:,2], 2)
    tips = pw[pw[:,2] < zmin + 0.02]
    return np.array([*((tips[:,:2].min(0)+tips[:,:2].max(0))/2), zmin]), tips[:,:2].mean(0)

FO = 0.1034
# empirical offset: observed_tip ~= commanded_tip + OFF (from previous run)
OFF = np.array([-0.007, -0.046, 0.111])

ce = cube_est()
ctr, zmax, zmin, th = ce
print("cube ctr", ctr, "zmax", zmax, "zmin", zmin, "yaw deg", np.degrees(th))
quat = np.array([0.0, np.cos(th/2), np.sin(th/2), 0.0])

def go(tip):
    j = solve_ik(np.asarray(tip, dtype=np.float64) + np.array([0, 0, FO]), quat)
    move_to_joints(j)

open_gripper()
corr = np.zeros(2)
# Hover with tips ~6cm above the cube top, then refine xy using gripper segmentation
for it in range(4):
    want = np.array([ctr[0], ctr[1], zmax + 0.06])
    cmd = want - OFF
    cmd[:2] += corr
    go(cmd)
    g = gripper_tip_est(ctr)
    if g is None:
        print("gripper not seen"); break
    tip, tipmean = g
    err = ctr - tip[:2]
    print(it, "cmd", np.round(cmd,3), "tip bbox ctr", np.round(tip,3), "tip mean", np.round(tipmean,3), "err", np.round(err,3))
    if np.linalg.norm(err) < 0.006:
        break
    if np.linalg.norm(err) > 0.08:
        print("error too large, ignoring"); break
    corr += 0.8 * err

# Descend
gx, gy = ctr[0] - OFF[0] + corr[0], ctr[1] - OFF[1] + corr[1]
target_tip_z = zmax - 0.025
cz = target_tip_z - OFF[2]
go([gx, gy, cz + 0.06])
go([gx, gy, cz + 0.03])
go([gx, gy, cz])
g = gripper_tip_est(ctr)
print("tip before close", None if g is None else np.round(g[0],3), "cube top", zmax)
close_gripper()
go([gx, gy, cz + 0.20])

ce2 = cube_est()
if ce2 is None:
    print("cube not found after lift")
else:
    print("after lift cube ctr", ce2[0], "zmax", ce2[1], "zmin", ce2[2], "before zmin", zmin)