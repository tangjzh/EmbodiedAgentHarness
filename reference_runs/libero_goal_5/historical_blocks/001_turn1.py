import numpy as np
d = np.array([0.12, 1.0, 0.0]); d = d / np.linalg.norm(d)
try:
    r = push("red-rimmed plate", direction=tuple(float(v) for v in d), distance=0.25)
    print("push result", r)
except Exception as e:
    print("push FAIL", e)
obs = get_observation()
print("tcp", np.round(np.array(obs["tcp_cartesian_pos"]),3))
for n in ["red-rimmed plate", "small blue box", "bowl"]:
    try:
        p = get_object_pose(n)
        print(n, np.round(np.array(p[0]),3))
    except Exception as e:
        print(n, "FAIL", e)
