import numpy as np, math

hx, hy_front, hz = 0.6789, -0.1261, 0.1133
q_side = [0.70710678, 0.70710678, 0.0, 0.0]
q_alt = [0.5, 0.5, -0.5, 0.5]

def tcp():
    return np.array(get_observation()["tcp_cartesian_pos"])

def go(pos, q, za=0.0):
    move_to([float(pos[0]), float(pos[1]), float(pos[2])], q, z_approach=za)
    cur = tcp()
    print("target", np.round(np.array(pos), 3), "reached", np.round(cur[:3], 3), "grip", round(float(cur[7]), 3))
    return cur

def attempt(q, dy, dz):
    open_gripper()
    go([hx, hy_front + 0.15, hz + dz + 0.10], q)
    go([hx, hy_front + 0.10, hz + dz], q)
    go([hx, hy_front + 0.05, hz + dz], q)
    go([hx, hy_front + dy, hz + dz], q)
    close_gripper()
    g = float(tcp()[7])
    print("grip after close", g)
    return g

grasped = False
used_q = q_side
used_dz = 0.0
trials = [(q_side, -0.015, 0.0), (q_alt, -0.015, 0.0), (q_side, -0.025, -0.008)]
for (q, dy, dz) in trials:
    g = attempt(q, dy, dz)
    if g > 0.03:
        grasped = True
        used_q = q
        used_dz = dz
        break
    open_gripper()
    go([hx, hy_front + 0.08, hz + dz], q)

if not grasped:
    print("no confirmed grasp; pulling with closed gripper hooked behind handle as fallback")
    open_gripper()
    go([hx, hy_front + 0.08, hz], q_side)
    go([hx, hy_front - 0.015, hz], q_side)
    close_gripper()
    used_q = q_side
    used_dz = 0.0

start = tcp()
sy = float(start[1])
sz = hz + used_dz
for k in range(1, 4):
    go([hx, sy + 0.06 * k, sz], used_q)
print("grip after pull", float(tcp()[7]))
open_gripper()
cur = tcp()
go([cur[0], cur[1] + 0.06, cur[2]], used_q)
go([cur[0], cur[1] + 0.06, cur[2] + 0.12], used_q)

obs = get_observation()
cam = obs["agentview"]
K = np.array(cam["intrinsics"]); T = np.array(cam["pose_mat"])
depth = np.array(cam["images"]["depth"]).squeeze()
R = T[:3, :3]; t = T[:3, 3]
def deproj(u, v):
    d = float(depth[v, u])
    pc = np.array([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d])
    return R @ pc + t
for v in range(110, 185, 6):
    row = [deproj(u, v) for u in range(60, 110, 4)]
    print(v, [round(float(p[1]), 3) for p in row])
