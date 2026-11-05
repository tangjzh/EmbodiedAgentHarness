import numpy as np

open_gripper()
obs = get_observation()
q = list(obs["tcp_cartesian_pos"][3:7])
c = np.array([0.67229744, 0.01500772, 0.0350146])

rim_z = 0.08
tgt = np.array([c[0], c[1] + 0.038, rim_z - 0.02])
print("target", tgt)
move_to(list(tgt), q, z_approach=0.12)

obs = get_observation()
print("final tcp", obs["tcp_cartesian_pos"])
try:
    p = get_object_pose("yellow and white mug")
    print("mug after", p)
except Exception as e:
    print("pose fail", e)
