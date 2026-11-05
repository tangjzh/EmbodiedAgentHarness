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

# Code block 6
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
    zmax = np.percentile(pw[:,2], 99)
    top = pw[pw[:,2] > zmax - 0.008]
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
    return R2.T @ cr, zmax, pw[:,2].min(), best_th, len(top)

def tip_est(near_xy):
    rgb, depth, K, T = cam_data()
    for prompt in ["robot gripper fingers", "robot gripper"]:
        res = segment_sam3_text_prompt(rgb, prompt)
        if not res:
            continue
        for b in sorted(res, key=lambda d: -d["score"]):
            pw = mask_to_world(b["mask"], depth, K, T)
            pw = pw[np.linalg.norm(pw[:,:2] - near_xy, axis=1) < 0.12]
            if len(pw) < 30:
                continue
            zmin = np.percentile(pw[:,2], 2)
            tips = pw[pw[:,2] < zmin + 0.015]
            return np.array([*tips[:,:2].mean(0), zmin])
    return None

FO = 0.1034
ctr, zmax, zmin, th, nt = cube_est()
print("cube ctr", ctr, "zmax", zmax, "zmin", zmin, "yaw", np.degrees(th), "ntop", nt)
quat = np.array([0.0, np.cos(th/2), np.sin(th/2), 0.0])

def go(tip):
    j = solve_ik(np.asarray(tip, dtype=np.float64) + np.array([0, 0, FO]), quat)
    move_to_joints(j)

open_gripper()
corr = np.zeros(2)
zoff = 0.0
for it in range(3):
    cmd = np.array([ctr[0] + corr[0], ctr[1] + corr[1], zmax + 0.08])
    go(cmd)
    t = tip_est(ctr)
    print(it, "cmd", np.round(cmd,3), "tip obs", None if t is None else np.round(t,3))
    if t is None:
        break
    err = cmd[:2] - t[:2]  # commanded minus observed position
    if np.linalg.norm(err) > 0.04:
        print("implausible err, ignoring", err); break
    zoff = t[2] - cmd[2] if abs(t[2] - cmd[2]) < 0.04 else 0.0
    newcorr = (ctr - t[:2]) + corr
    if np.linalg.norm(ctr - t[:2]) < 0.005:
        break
    corr = newcorr
print("corr", corr, "zoff", zoff)

gx, gy = ctr[0] + corr[0], ctr[1] + corr[1]
tz = zmax - 0.02 - zoff
for dz in [0.05, 0.025, 0.0]:
    go([gx, gy, tz + dz])
print("tip before close", tip_est(ctr))
close_gripper()
go([gx, gy, tz + 0.2])

ce2 = cube_est()
print("after lift", None if ce2 is None else (ce2[0], ce2[1], ce2[2]), "before zmin", zmin)

# Code block 7
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
    zmax = np.percentile(pw[:,2], 99)
    top = pw[pw[:,2] > zmax - 0.008]
    ctr = (top[:,:2].min(0) + top[:,:2].max(0)) / 2
    return ctr, zmax, pw[:,2].min()

def tip_est(near_xy):
    rgb, depth, K, T = cam_data()
    for prompt in ["robot gripper fingers", "robot gripper"]:
        res = segment_sam3_text_prompt(rgb, prompt)
        if not res:
            continue
        for b in sorted(res, key=lambda d: -d["score"]):
            pw = mask_to_world(b["mask"], depth, K, T)
            pw = pw[np.linalg.norm(pw[:,:2] - near_xy, axis=1) < 0.12]
            if len(pw) < 30:
                continue
            zmin = np.percentile(pw[:,2], 2)
            tips = pw[pw[:,2] < zmin + 0.015]
            return np.array([*tips[:,:2].mean(0), zmin])
    return None

FO = 0.1034
th = np.radians(-3.0)
quat = np.array([0.0, np.cos(th/2), np.sin(th/2), 0.0])
def go(tip):
    j = solve_ik(np.asarray(tip, dtype=np.float64) + np.array([0, 0, FO]), quat)
    move_to_joints(j)

ctr, zmax, zmin = cube_est()
print("cube", ctr, zmax, zmin)

# Measured consistent offset: observed_tip = cmd + OFF
OFF = np.array([-0.003, -0.045, 0.097])

open_gripper()
# Hover: tips 6cm above cube top
want = np.array([ctr[0], ctr[1], zmax + 0.06])
cmd = want - OFF
go(cmd)
t = tip_est(ctr)
print("hover cmd", np.round(cmd,3), "tip", None if t is None else np.round(t,3), "want", np.round(want,3))
if t is not None and np.linalg.norm(want - t) < 0.05:
    cmd = cmd + (want - t)
    go(cmd)
    t = tip_est(ctr)
    print("refined hover tip", None if t is None else np.round(t,3))
    if t is not None and np.linalg.norm(want - t) < 0.03:
        cmd = cmd + 0.8*(want - t)

# Descend to grasp height (tips ~2cm below cube top)
grasp_cmd = cmd + np.array([0, 0, -0.08])
go(cmd + np.array([0, 0, -0.04]))
t = tip_est(ctr)
want2 = np.array([ctr[0], ctr[1], zmax + 0.02])
print("mid tip", None if t is None else np.round(t,3), "want", np.round(want2,3))
if t is not None and np.linalg.norm(want2[:2] - t[:2]) < 0.02:
    grasp_cmd[:2] += 0.8*(want2[:2] - t[:2])
go(grasp_cmd)
print("tip before close", tip_est(ctr), "cube top", zmax)
close_gripper()
go(grasp_cmd + np.array([0, 0, 0.2]))

ce2 = cube_est()
print("after lift", ce2, "before zmin", zmin)
if ce2 is not None and ce2[2] > zmin + 0.05:
    print("SUCCESS: cube lifted")
else:
    print("cube not lifted")

# Code block 8
import numpy as np

def cam_data():
    obs = get_observation()
    cam = obs["robot0_robotview"]
    return cam["images"]["rgb"], cam["images"]["depth"].astype(np.float64), cam["intrinsics"], cam["pose_mat"]

def pix_world(ys, xs, depth, K, T):
    d = depth[ys, xs]
    pc = np.stack([(xs-K[0,2])*d/K[0,0], (ys-K[1,2])*d/K[1,1], d], 1)
    return (T[:3,:3] @ pc.T).T + T[:3,3]

def cube_est():
    rgb, depth, K, T = cam_data()
    res = segment_sam3_text_prompt(rgb, "red cube")
    b = max(res, key=lambda d: d["score"])
    ys, xs = np.nonzero(b["mask"])
    v = np.isfinite(depth[ys, xs]) & (depth[ys, xs] > 0)
    pw = pix_world(ys[v], xs[v], depth, K, T)
    zmax = np.percentile(pw[:,2], 99)
    top = pw[pw[:,2] > zmax - 0.008]
    best_a, best_th = 1e9, 0.0
    for th in np.linspace(-np.pi/4, np.pi/4, 91):
        c, s = np.cos(th), np.sin(th)
        R2 = np.array([[c, s], [-s, c]])
        p = top[:, :2] @ R2.T
        a = np.ptp(p[:,0]) * np.ptp(p[:,1])
        if a < best_a: best_a, best_th = a, th
    c, s = np.cos(best_th), np.sin(best_th)
    R2 = np.array([[c, s], [-s, c]])
    p = top[:, :2] @ R2.T
    cr = np.array([(p[:,0].min()+p[:,0].max())/2, (p[:,1].min()+p[:,1].max())/2])
    return R2.T @ cr, zmax, pw[:,2].min(), best_th

ctr, zmax, zmin, th = cube_est()
print("cube", ctr, zmax, zmin, np.degrees(th))
quat = np.array([0.0, np.cos(th/2), np.sin(th/2), 0.0])
FO = 0.1034
def go(tip):
    j = solve_ik(np.asarray(tip, dtype=np.float64) + np.array([0, 0, FO]), quat)
    move_to_joints(j)

# Baseline depth: arm high, far from the cube region
open_gripper()
go([ctr[0] - 0.15, ctr[1] + 0.25, zmax + 0.35])
_, D0, K, T = cam_data()

def tip_by_diff():
    _, D1, K, T = cam_data()
    m = np.isfinite(D1) & (D1 > 0) & np.isfinite(D0) & (D1 < D0 - 0.015)
    ys, xs = np.nonzero(m)
    if len(ys) < 50: return None
    pw = pix_world(ys, xs, D1, K, T)
    pw = pw[pw[:,2] > zmin + 0.005]
    pw = pw[np.linalg.norm(pw[:,:2] - ctr, axis=1) < 0.2]
    if len(pw) < 30: return None
    zl = np.percentile(pw[:,2], 1)
    tips = pw[pw[:,2] < zl + 0.015]
    return np.array([*np.median(tips[:,:2], 0), zl])

cmd = np.array([ctr[0], ctr[1], zmax + 0.10])
want = cmd.copy()
off = np.zeros(3)
for it in range(4):
    go(cmd)
    t = tip_by_diff()
    print(it, "cmd", np.round(cmd,3), "tip", None if t is None else np.round(t,3), "want", np.round(want,3))
    if t is None: break
    e = want - t
    if np.linalg.norm(e) > 0.15: print("implausible"); break
    off = (t - cmd)
    if np.linalg.norm(e) < 0.005: break
    cmd = cmd + 0.9*e

# observed = cmd + off  -> cmd = desired - off
grasp_tip = np.array([ctr[0], ctr[1], zmax - 0.02])
gcmd = grasp_tip - off
print("offset", np.round(off,3), "grasp cmd", np.round(gcmd,3))
go(gcmd + np.array([0,0,0.06]))
t = tip_by_diff()
print("pre tip", t, "want", grasp_tip + np.array([0,0,0.06]))
if t is not None:
    e = grasp_tip + np.array([0,0,0.06]) - t
    if np.linalg.norm(e[:2]) < 0.03:
        gcmd[:2] += 0.9*e[:2]
go(gcmd + np.array([0,0,0.03]))
go(gcmd)
close_gripper()
go(gcmd + np.array([0,0,0.2]))

c2, zmax2, zmin2, _ = cube_est()
print("after", c2, zmax2, zmin2, "before zmin", zmin)
print("SUCCESS" if zmin2 > zmin + 0.05 else "not lifted")