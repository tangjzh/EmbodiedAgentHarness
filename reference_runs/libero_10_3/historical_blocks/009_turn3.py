import numpy as np
obs = get_observation()
quat = list(obs["tcp_cartesian_pos"][3:7])
GRIP = 0.0
start = np.array(obs["tcp_cartesian_pos"][:3])
target = np.array([0.625, 0.165, start[2]])
q = None
n = 6
for i in range(1, n+1):
    p = start + (target - start) * i / n
    q = solve_ik(list(p), quat)
    if q is None:
        break
    command_joints(list(q), GRIP, steps=6)
if q is not None:
    command_joints(list(q), GRIP, steps=20)
print("final tcp", get_observation()["tcp_cartesian_pos"])
