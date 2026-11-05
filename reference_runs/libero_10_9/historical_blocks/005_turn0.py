import numpy as np
obs = get_observation()
tcp = obs["tcp_cartesian_pos"]
print("tcp", tcp)
res = {}
for name in ["microwave", "yellow and white mug", "white mug"]:
    try:
        p = get_object_pose(name)
        res[name] = p
        print(name, p)
    except Exception as e:
        print(name, "ERR", e)
# lift mug a bit for clearance, keep orientation
pos = list(tcp[:3]); q = list(tcp[3:7])
pos[2] = max(pos[2], 0.30)
move_to(pos, q)
obs = get_observation()
print("after lift tcp", obs["tcp_cartesian_pos"])
CONTEXT["poses"] = {k: (list(map(float, v[0])), list(map(float, v[1]))) for k, v in res.items() if v is not None}
