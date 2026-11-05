import numpy as np
obs = get_observation()
tcp = np.array(obs["tcp_cartesian_pos"], dtype=float)
q = [float(v) for v in tcp[3:7]]
av = obs["agentview"]
try:
    print("agentview keys", list(av.keys()))
except Exception:
    print("agentview type", type(av))
info = {}
for name in ["microwave door", "inside of the microwave", "yellow and white mug"]:
    try:
        p, o = get_object_pose(name)
        print(name, p, o)
        if p is not None:
            info[name] = [float(v) for v in p]
    except Exception as e:
        print(name, "ERR", e)
CONTEXT["info"] = info
mw = np.array(CONTEXT["mw"], dtype=float)
# pre-entry: move in front of the side opening, outside microwave footprint
x_t = float(np.clip(mw[0], 0.72, 0.82))
move_to([x_t, float(tcp[1]), 0.34], q)
o1 = get_observation()["tcp_cartesian_pos"]
print("pre1", o1)
move_to([x_t, float(tcp[1]), 0.27], q)
o2 = get_observation()["tcp_cartesian_pos"]
print("pre2", o2)
CONTEXT["pre"] = [float(v) for v in o2[:3]]
try:
    p, o = get_object_pose("yellow and white mug")
    print("mug after", p)
except Exception as e:
    print("mug ERR", e)
