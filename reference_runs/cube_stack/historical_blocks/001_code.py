# Code block 0
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"].astype(np.float64)
K = cam["intrinsics"]
T_wc = cam["pose_mat"]
H, W = depth.shape

def get_mask(prompt, fallback_pt):
    res = segment_sam3_text_prompt(rgb, prompt)
    if len(res) > 0:
        best = max(res, key=lambda d: d["score"])
        return best["mask"].astype(bool)
    res = segment_sam3_point_prompt(rgb, fallback_pt)
    best = max(res, key=lambda d: d["score"])
    return best["mask"].astype(bool)

def mask_to_world(mask):
    v, u = np.nonzero(mask)
    z = depth[v, u]
    ok = np.isfinite(z) & (z > 0.05) & (z < 5.0)
    u, v, z = u[ok], v[ok], z[ok]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z, np.ones_like(z)], axis=1)
    pw = (T_wc @ pc.T).T[:, :3]
    return pw

def cube_info(pw):
    top_z = np.percentile(pw[:, 2], 97)
    bot_z = np.percentile(pw[:, 2], 2)
    # use top-face points for xy center and yaw
    top_pts = pw[pw[:, 2] > top_z - 0.01]
    if len(top_pts) < 20:
        top_pts = pw
    cxy = np.median(top_pts[:, :2], axis=0)
    # yaw from minimum-area rectangle search
    xy = top_pts[:, :2] - cxy
    best_yaw, best_area = 0.0, 1e9
    for a in np.linspace(0, np.pi / 2, 90, endpoint=False):
        c, s = np.cos(a), np.sin(a)
        r = xy @ np.array([[c, -s], [s, c]])
        area = np.ptp(r[:, 0]) * np.ptp(r[:, 1])
        if area < best_area:
            best_area, best_yaw = area, a
    return cxy, top_z, bot_z, best_yaw

red_mask = get_mask("red cube", (280, 245))
green_mask = get_mask("green cube", (170, 432))
red_pw = mask_to_world(red_mask)
green_pw = mask_to_world(green_mask)

red_xy, red_top, red_bot, red_yaw = cube_info(red_pw)
green_xy, green_top, green_bot, green_yaw = cube_info(green_pw)
red_h = max(red_top - red_bot, 0.02)
print("red", red_xy, red_top, red_bot, red_yaw)
print("green", green_xy, green_top, green_bot, green_yaw)

def mat_to_quat_wxyz(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

def down_quat(yaw):
    # pick yaw equivalent within [-pi/4, pi/4] to avoid wrist wrap
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    return mat_to_quat_wxyz(Rz @ Rx)

TCP = 0.1034  # panda_hand to fingertip center

def move_tip(p, q):
    j = solve_ik(np.array([p[0], p[1], p[2] + TCP]), q)
    move_to_joints(j)

q_red = down_quat(red_yaw)
q_green = down_quat(green_yaw)

grip_depth = min(0.02, red_h * 0.5)
grasp_z = red_top - grip_depth

open_gripper()
move_tip([red_xy[0], red_xy[1], red_top + 0.15], q_red)
move_tip([red_xy[0], red_xy[1], red_top + 0.05], q_red)
move_tip([red_xy[0], red_xy[1], grasp_z], q_red)
close_gripper()
move_tip([red_xy[0], red_xy[1], red_top + 0.15], q_red)

# transport to above green cube
place_tip_z = green_top + (red_h - grip_depth) + 0.012
move_tip([green_xy[0], green_xy[1], place_tip_z + 0.12], q_green)
move_tip([green_xy[0], green_xy[1], place_tip_z + 0.03], q_green)
move_tip([green_xy[0], green_xy[1], place_tip_z], q_green)
open_gripper()
move_tip([green_xy[0], green_xy[1], place_tip_z + 0.12], q_green)

# Code block 1
import numpy as np

def observe():
    obs = get_observation()
    cam = obs["robot0_robotview"]
    return cam["images"]["rgb"], cam["images"]["depth"].astype(np.float64), cam["intrinsics"], cam["pose_mat"]

def get_mask_img(rgb, prompt, fallback_pt):
    res = segment_sam3_text_prompt(rgb, prompt)
    if len(res) > 0:
        return max(res, key=lambda d: d["score"])["mask"].astype(bool)
    res = segment_sam3_point_prompt(rgb, fallback_pt)
    return max(res, key=lambda d: d["score"])["mask"].astype(bool)

def mask_to_world2(mask, depth, K, T_wc):
    v, u = np.nonzero(mask)
    z = depth[v, u]
    ok = np.isfinite(z) & (z > 0.05) & (z < 5.0)
    u, v, z = u[ok], v[ok], z[ok]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z, np.ones_like(z)], axis=1)
    return (T_wc @ pc.T).T[:, :3]

# move up and away first so the arm doesn't occlude
open_gripper()
move_tip([0.5, 0.0, 0.25], down_quat(0.0))

rgb, depth, K, T_wc = observe()
red_mask = get_mask_img(rgb, "red cube", (280, 245))
green_mask = get_mask_img(rgb, "green cube", (168, 432))
red_xy, red_top, red_bot, red_yaw = cube_info(mask_to_world2(red_mask, depth, K, T_wc))
green_xy, green_top, green_bot, green_yaw = cube_info(mask_to_world2(green_mask, depth, K, T_wc))
red_h = float(np.clip(red_top - red_bot, 0.03, 0.06))
print("red", red_xy, red_top, red_bot, red_yaw)
print("green", green_xy, green_top, green_bot, green_yaw)

q_red = down_quat(red_yaw)
q_green = down_quat(green_yaw)

# grasp deeper: around mid-height of the cube
grip_depth = red_h * 0.6
grasp_z = red_top - grip_depth

move_tip([red_xy[0], red_xy[1], red_top + 0.15], q_red)
move_tip([red_xy[0], red_xy[1], red_top + 0.05], q_red)
move_tip([red_xy[0], red_xy[1], grasp_z], q_red)
close_gripper()
move_tip([red_xy[0], red_xy[1], red_top + 0.15], q_red)

# verify the red cube has been lifted
rgb2, depth2, _, _ = observe()
res = segment_sam3_text_prompt(rgb2, "red cube")
if len(res) > 0:
    m = max(res, key=lambda d: d["score"])["mask"].astype(bool)
    pw = mask_to_world2(m, depth2, K, T_wc)
    if len(pw) > 0:
        print("red after lift median z:", np.median(pw[:, 2]))

place_tip_z = green_top + (red_h - grip_depth) + 0.01
move_tip([green_xy[0], green_xy[1], place_tip_z + 0.12], q_green)
move_tip([green_xy[0], green_xy[1], place_tip_z + 0.03], q_green)
move_tip([green_xy[0], green_xy[1], place_tip_z], q_green)
open_gripper()
move_tip([green_xy[0], green_xy[1], place_tip_z + 0.12], q_green)

# Code block 2
import numpy as np

def observe():
    obs = get_observation()
    cam = obs["robot0_robotview"]
    return cam["images"]["rgb"], cam["images"]["depth"].astype(np.float64), cam["intrinsics"], cam["pose_mat"]

def best_mask(rgb, prompt):
    res = segment_sam3_text_prompt(rgb, prompt)
    if len(res) == 0:
        return None
    return max(res, key=lambda d: d["score"])["mask"].astype(bool)

# Fresh observation of the cubes with the arm out of the way
open_gripper()
move_tip([0.45, 0.25, 0.25], down_quat(0.0))
rgb, depth, K, T_wc = observe()
rm = best_mask(rgb, "red cube")
gm = best_mask(rgb, "green cube")
red_xy, red_top, red_bot, red_yaw = cube_info(mask_to_world2(rm, depth, K, T_wc))
green_xy, green_top, green_bot, green_yaw = cube_info(mask_to_world2(gm, depth, K, T_wc))
red_h = float(np.clip(red_top - red_bot, 0.03, 0.06))
print("red", red_xy, red_top, red_bot, red_yaw)
print("green", green_xy, green_top, green_bot, green_yaw)
q_red = down_quat(red_yaw)
q_green = down_quat(green_yaw)

# Calibration: move fingertips to a known point above red cube and measure where gripper actually is
target = np.array([red_xy[0], red_xy[1], red_top + 0.10])
move_tip(target, q_red)
rgb_c, depth_c, K, T_wc = observe()
gmask = best_mask(rgb_c, "robot gripper fingers")
dxy = np.zeros(2)
dz = 0.0
if gmask is not None:
    gp = mask_to_world2(gmask, depth_c, K, T_wc)
    if len(gp) > 50:
        zmin = np.percentile(gp[:, 2], 2)
        low = gp[gp[:, 2] < zmin + 0.015]
        meas_xy = np.median(low[:, :2], axis=0)
        print("gripper measured tip:", meas_xy, zmin, "target:", target)
        dz_c = zmin - target[2]
        dxy_c = meas_xy - target[:2]
        # only apply plausible corrections
        if abs(dz_c) < 0.08:
            dz = dz_c
        if np.linalg.norm(dxy_c) < 0.04:
            dxy = dxy_c
print("correction dxy, dz:", dxy, dz)

def move_tip_c(p, q):
    p = np.array(p, dtype=float)
    move_tip([p[0] - dxy[0], p[1] - dxy[1], p[2] - dz], q)

grip_depth = red_h * 0.6
grasp_z = max(red_top - grip_depth, red_bot + 0.008)

move_tip_c([red_xy[0], red_xy[1], red_top + 0.05], q_red)
move_tip_c([red_xy[0], red_xy[1], grasp_z], q_red)
close_gripper()
move_tip_c([red_xy[0], red_xy[1], red_top + 0.15], q_red)

rgb2, depth2, K, T_wc = observe()
m = best_mask(rgb2, "red cube")
lifted = False
if m is not None:
    pw = mask_to_world2(m, depth2, K, T_wc)
    if len(pw) > 0:
        mz = np.median(pw[:, 2])
        print("red after lift median z:", mz)
        lifted = mz > red_top + 0.03
print("lifted:", lifted)

if lifted:
    place_tip_z = green_top + (red_top - grasp_z) + 0.01
    move_tip_c([green_xy[0], green_xy[1], place_tip_z + 0.12], q_green)
    move_tip_c([green_xy[0], green_xy[1], place_tip_z + 0.03], q_green)
    move_tip_c([green_xy[0], green_xy[1], place_tip_z], q_green)
    open_gripper()
    move_tip_c([green_xy[0], green_xy[1], place_tip_z + 0.12], q_green)
else:
    open_gripper()
    move_tip_c([red_xy[0], red_xy[1], red_top + 0.15], q_red)

# Code block 3
import numpy as np

def observe():
    obs = get_observation()
    cam = obs["robot0_robotview"]
    return cam["images"]["rgb"], cam["images"]["depth"].astype(np.float64), cam["intrinsics"], cam["pose_mat"]

def best_mask(rgb, prompt):
    res = segment_sam3_text_prompt(rgb, prompt)
    if len(res) == 0:
        return None
    return max(res, key=lambda d: d["score"])["mask"].astype(bool)

def move_hand(p, q):
    move_to_joints(solve_ik(np.asarray(p, dtype=float), q))

# Move arm out of view and observe
open_gripper()
move_tip([0.45, 0.25, 0.25], down_quat(0.0))
rgb, depth, K, T_wc = observe()
rm = best_mask(rgb, "red cube")
gm = best_mask(rgb, "green cube")
red_xy, red_top, red_bot, red_yaw = cube_info(mask_to_world2(rm, depth, K, T_wc))
green_xy, green_top, green_bot, green_yaw = cube_info(mask_to_world2(gm, depth, K, T_wc))
print("red", red_xy, red_top, red_bot)
print("green", green_xy, green_top, green_bot)

# Plan grasps with Contact-GraspNet on the red cube
seg = rm.astype(np.int32)
grasp_T = None
try:
    poses, scores = plan_grasp(depth.astype(np.float32), K, seg)
    print("num grasps:", len(scores))
    best_s = -1
    for T, s in zip(poses, scores):
        Tw = T_wc @ T
        approach = Tw[:3, 2]
        # prefer top-down grasps
        if approach[2] < -0.8:
            sc = s + 0.5 * (-approach[2])
            if sc > best_s:
                best_s = sc
                grasp_T = Tw
except Exception as e:
    print("plan_grasp failed:", e)

if grasp_T is not None:
    R = grasp_T[:3, :3]
    q_g = mat_to_quat_wxyz(R)
    hand_pos = grasp_T[:3, 3].copy()
    approach = R[:, 2]
    print("graspnet hand pos:", hand_pos, "approach:", approach)
    # sanity: fingertip position should be near red cube
    tip = hand_pos + approach * TCP
    if np.linalg.norm(tip[:2] - red_xy) > 0.04 or tip[2] > red_top + 0.01 or tip[2] < red_bot - 0.01:
        print("graspnet grasp inconsistent, using heuristic. tip:", tip)
        grasp_T = None

if grasp_T is None:
    q_g = down_quat(red_yaw)
    approach = np.array([0, 0, -1.0])
    grasp_z = max(red_top - 0.6 * (red_top - red_bot), red_bot + 0.01)
    hand_pos = np.array([red_xy[0], red_xy[1], grasp_z + TCP])

pre = hand_pos - approach * 0.10
move_hand(pre, q_g)
move_hand(hand_pos - approach * 0.04, q_g)
move_hand(hand_pos, q_g)
close_gripper()
lift = hand_pos.copy(); lift[2] += 0.15
move_hand(lift, q_g)

rgb2, depth2, K, T_wc = observe()
m = best_mask(rgb2, "red cube")
lifted = False
if m is not None:
    pw = mask_to_world2(m, depth2, K, T_wc)
    if len(pw) > 0:
        mz = np.median(pw[:, 2])
        print("red after lift median z:", mz)
        lifted = mz > red_top + 0.03
print("lifted:", lifted)

if lifted:
    offset_xy = hand_pos[:2] - red_xy
    d = hand_pos[2] - red_bot
    place = np.array([green_xy[0] + offset_xy[0], green_xy[1] + offset_xy[1], green_top + d + 0.01])
    above = place.copy(); above[2] += 0.12
    move_hand(above, q_g)
    mid = place.copy(); mid[2] += 0.03
    move_hand(mid, q_g)
    move_hand(place, q_g)
    open_gripper()
    move_hand(above, q_g)
else:
    open_gripper()
    move_hand(lift, q_g)