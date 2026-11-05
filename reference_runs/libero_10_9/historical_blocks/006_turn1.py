import numpy as np
obs = get_observation()
print("keys", list(obs.keys()) if hasattr(obs, "keys") else type(obs))
tcp = np.array(obs["tcp_cartesian_pos"], dtype=float)
q = [float(v) for v in tcp[3:7]]
found = {}
for name in ["microwave oven", "black microwave", "microwave", "microwave door"]:
    try:
        p, o = get_object_pose(name)
        print(name, p, o)
        if p is not None:
            found[name] = (np.array(p, dtype=float), np.array(o, dtype=float))
            if name != "microwave door":
                break
    except Exception as e:
        print(name, "ERR", e)
mw = None
for k in ["microwave oven", "black microwave", "microwave"]:
    if k in found:
        mw = found[k]; break
CONTEXT["mw"] = None if mw is None else [float(v) for v in mw[0]]
CONTEXT["mw_q"] = None if mw is None else [float(v) for v in mw[1]]
if "microwave door" in found:
    CONTEXT["door"] = [float(v) for v in found["microwave door"][0]]
if mw is not None:
    m = mw[0]
    print("microwave pos", m)
    d = tcp[:2] - m[:2]
    print("dist xy to tcp", np.linalg.norm(d))
    # rotation matrix of microwave pose for inspection
    w,x,y,z = mw[1]
    R = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                  [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                  [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    print("R", R)
    # move above the microwave front region at safe height (no entry yet)
    target = [float(tcp[0]), float(tcp[1]), float(max(tcp[2], 0.33))]
    move_to(target, q)
print("tcp now", get_observation()["tcp_cartesian_pos"])
