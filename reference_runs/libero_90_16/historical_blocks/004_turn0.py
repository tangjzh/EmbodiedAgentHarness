import numpy as np

obs = get_observation()
tcp = np.array(obs["tcp_cartesian_pos"][:3], dtype=float)
quat = list(obs["tcp_cartesian_pos"][3:7])
q_cur = np.array(obs["robot_joint_pos"][:7], dtype=float)
start_z = tcp[2]
target_z = start_z + 0.13

n_wp = 13
prev = q_cur.copy()
for i in range(1, n_wp + 1):
    z = start_z + (target_z - start_z) * i / n_wp
    pos = [tcp[0], tcp[1], z]
    q = solve_ik(pos, quat)
    if q is None:
        print("IK failed at z", z)
        break
    q = np.array(q, dtype=float)[:7]
    if np.max(np.abs(q - prev)) > 0.5:
        print("IK jump too large at z", z, np.max(np.abs(q - prev)))
        break
    command_joints(q.tolist(), 0.0, steps=12)
    prev = q

# settle
command_joints(prev.tolist(), 0.0, steps=20)
obs = get_observation()
print("start tcp z", start_z)
print("final tcp", obs["tcp_cartesian_pos"])
print("lift", obs["tcp_cartesian_pos"][2] - start_z)
