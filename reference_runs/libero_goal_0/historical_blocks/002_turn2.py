import numpy as np, math

obs = get_observation()
cam = obs["agentview"]
K = np.array(cam["intrinsics"]); T = np.array(cam["pose_mat"])
depth = np.array(cam["images"]["depth"]).squeeze()
R = T[:3, :3]; t = T[:3, 3]

def deproj(u, v):
    d = float(depth[v, u])
    pc = np.array([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d])
    return R @ pc + t

def handle_point(u, v):
    pts = np.array([deproj(uu, vv) for vv in range(v - 1, v + 2) for uu in range(u - 1, u + 2)])
    center = np.median(pts, axis=0)
    win = []
    for vv in range(v - 4, v + 5):
        for uu in range(u - 6, u + 7):
            p = deproj(uu, vv)
            if p[2] > 0.02 and abs(p[2] - center[2]) < 0.03 and abs(p[0] - center[0]) < 0.06:
                win.append(p)
    win = np.array(win) if len(win) else pts
    ymax = np.percentile(win[:, 1], 90)
    return center, ymax

for name, (u, v) in {"top": (79, 121), "mid": (80, 147), "bot": (80, 171)}.items():
    c, ym = handle_point(u, v)
    print(name, "center", np.round(c, 4), "ymax", round(float(ym), 4))

c, ym = handle_point(80, 147)
p = c.copy()
if abs(ym - c[1]) < 0.04:
    p[1] = max(c[1], ym)
# sanity fallback to perception estimate
if not (0.45 < p[0] < 0.9 and -0.35 < p[1] < 0.05 and -0.02 < p[2] < 0.25):
    print("depth estimate implausible, using fallback")
    p = np.array([0.67, -0.137, 0.08])
print("middle handle target", p)

q_side = [0.70710678, 0.70710678, 0.0, 0.0]   # tool z -> -y, fingers close along z
hx, hy, hz = p

def tcp():
    return np.array(get_observation()["tcp_cartesian_pos"])

def go(pos, q=q_side, za=0.0):
    move_to(list(map(float, pos)), q, z_approach=za)
    cur = tcp()
    print("target", np.round(pos, 3), "reached", np.round(cur[:3], 3), "grip", round(float(cur[7]), 3))
    return cur

open_gripper()
# raise and reorient away from cabinet
go([hx, hy + 0.18, max(hz + 0.12, 0.2)])
go([hx, hy + 0.12, hz])
go([hx, hy + 0.06, hz])
cur = go([hx, hy - 0.012, hz])
close_gripper()
g = tcp()[7]
print("grip after close", g)

if g < 0.02:
    print("missed grasp, retrying slightly deeper/lower")
    open_gripper()
    go([hx, hy + 0.06, hz])
    for dz in [-0.012, 0.012]:
        go([hx, hy + 0.04, hz + dz])
        go([hx, hy - 0.02, hz + dz])
        close_gripper()
        g = tcp()[7]
        print("retry grip", dz, g)
        if g >= 0.02:
            hz = hz + dz
            break
        open_gripper()
        go([hx, hy + 0.06, hz + dz])

# pull outward along +y in small steps
start_y = tcp()[1]
for k in range(1, 7):
    go([hx, start_y + 0.03 * k, hz])
print("grip after pull", tcp()[7])
open_gripper()
cur = tcp()
go([cur[0], cur[1] + 0.06, cur[2]])
go([cur[0], cur[1] + 0.06, cur[2] + 0.12])

# re-check middle handle location
obs = get_observation()
cam = obs["agentview"]; depth = np.array(cam["images"]["depth"]).squeeze()
for name, (u, v) in {"top": (79, 121), "mid": (80, 147), "bot": (80, 171)}.items():
    c, ym = handle_point(u, v)
    print("after", name, np.round(c, 4), round(float(ym), 4))
try:
    print("pose middle handle after", get_object_pose("middle drawer handle"))
except Exception as e:
    print("pose err", e)
