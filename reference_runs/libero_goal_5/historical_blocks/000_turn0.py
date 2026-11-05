import numpy as np
names = ["red-rimmed plate", "stove", "bowl", "wine bottle", "small blue box"]
poses = {}
for n in names:
    try:
        p = get_object_pose(n)
        poses[n] = p
        print(n, np.round(np.array(p[0]),3), np.round(np.array(p[1]),3))
    except Exception as e:
        print(n, "FAIL", e)
obs = get_observation()
print("tcp", np.round(np.array(obs["tcp_cartesian_pos"]),3))
