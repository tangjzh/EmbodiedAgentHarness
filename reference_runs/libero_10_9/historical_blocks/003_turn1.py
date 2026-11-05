import numpy as np
obs = get_observation()
tcp = np.array(obs["tcp_cartesian_pos"])
q = tcp[3:7].tolist()

cx, cy = 0.675, 0.021          # estimated mug center
target = [cx + 0.003, cy - 0.042, 0.072]   # on the -y wall, fingers straddling rim

open_gripper()
# pre-approach above
move_to([target[0], target[1], 0.17], q)
o = get_observation(); print("above:", np.round(o["tcp_cartesian_pos"],4))
move_to(target, q, z_approach=0.08)
o = get_observation(); print("at grasp:", np.round(o["tcp_cartesian_pos"],4))
close_gripper()
o = get_observation()
print("after close tcp:", np.round(o["tcp_cartesian_pos"],4))
print("joints:", np.round(o["robot_joint_pos"],4))
