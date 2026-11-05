import numpy as np

obs = get_observation()
tcp = np.array(obs["tcp_cartesian_pos"], dtype=float)
start_pos = tcp[:3].copy()
quat = tcp[3:7].copy()
start_z = start_pos[2]

target_z = start_z + 0.16
n_way = 8
prev_q = np.array(obs["robot_joint_pos"][:7], dtype=float)
for i in range(1, n_way + 1):
    z = start_z + (target_z - start_z) * i / n_way
    p = [start_pos[0], start_pos[1], z]
    q = solve_ik(p, quat.tolist())
    if q is None:
        print("IK failed at waypoint", i)
        break
    q = np.array(q, dtype=float)
    # interpolate joints in small chunks
    for k in range(1, 4):
        qq = prev_q + (q - prev_q) * k / 3.0
        command_joints(qq.tolist(), 0.0, steps=6)
    prev_q = q

# settle
command_joints(prev_q.tolist(), 0.0, steps=20)

obs = get_observation()
tcp2 = obs["tcp_cartesian_pos"]
print("start z", start_z, "final tcp", tcp2[:3], "gripper", tcp2[7], "lift", tcp2[2] - start_z)
