import numpy as np
obs = get_observation()
if isinstance(obs, dict):
    for k, v in obs.items():
        try:
            a = np.array(v)
            print(k, a.shape if a.size > 20 else v)
        except Exception:
            print(k, type(v))
    print("tcp", obs.get("tcp_cartesian_pos"))
names = ["cabinet", "middle drawer handle", "middle handle", "drawer handle", "top drawer handle", "bottom drawer handle"]
for name in names:
    try:
        r = get_object_pose(name)
        print("POSE", name, r)
    except Exception as e:
        print("POSE", name, "ERR", e)
try:
    g = sample_grasp_pose("middle drawer handle")
    print("GRASP middle drawer handle", g)
except Exception as e:
    print("GRASP ERR", e)