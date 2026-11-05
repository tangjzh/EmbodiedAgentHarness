import numpy as np
obs = get_observation()
print("tcp", obs.get("tcp_cartesian_pos") if isinstance(obs, dict) else None)
if isinstance(obs, dict):
    print("keys", list(obs.keys()))
results = {}
for name in ["flat_stove_1", "flat_stove", "stove", "stove knob", "flat_stove_1_knob"]:
    try:
        p = get_object_pose(name)
        results[name] = p
        print(name, p)
    except Exception as e:
        print(name, "ERR", e)
