import numpy as np

obs = get_observation()
tcp = np.array(obs["tcp_cartesian_pos"][:3])
quat = list(obs["tcp_cartesian_pos"][3:7])

bowl_mid = np.array(CONTEXT["public_context"]["bowl_poses"]["akita_black_bowl_1"]) if "public_context" in CONTEXT else np.array([0.5957, 0.2097, 0.0186])
grasp_pos = np.array([0.757955549577344, 0.1794708572938487, 0.035])
front_center = np.array([0.7, 0.195])
offset_xy = grasp_pos[:2] - front_center  # TCP offset relative to held bowl center

target = np.array([bowl_mid[0] + offset_xy[0], bowl_mid[1] + offset_xy[1], 0.145])

# First ensure enough lift, then move horizontally
start = tcp.copy()
lift = start.copy(); lift[2] = max(start[2], 0.16)
waypoints = []
for a in np.linspace(0, 1, 4)[1:]:
    waypoints.append(start + a * (lift - start))
for a in np.linspace(0, 1, 10)[1:]:
    p = lift + a * (target - lift)
    p[2] = lift[2]
    waypoints.append(p)
final = target.copy()
for a in np.linspace(0, 1, 3)[1:]:
    p = waypoints[-1] * (1 - a) + final * a
    waypoints.append(p)

for p in waypoints:
    q = solve_ik(list(p), quat)
    if q is None:
        continue
    command_joints(list(q), 0.0, steps=12)

q = solve_ik(list(final), quat)
if q is not None:
    command_joints(list(q), 0.0, steps=30)

obs = get_observation()
print("tcp", obs["tcp_cartesian_pos"])
print("target", target, "mid bowl", bowl_mid)
