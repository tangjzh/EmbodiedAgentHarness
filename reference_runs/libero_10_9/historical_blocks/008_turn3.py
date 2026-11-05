import numpy as np
obs = get_observation()
tcp = np.array(obs["tcp_cartesian_pos"], dtype=float)
q = [float(v) for v in tcp[3:7]]
x_t = float(tcp[0])
z_t = 0.19

def go(p):
    move_to([float(p[0]), float(p[1]), float(p[2])], q)
    o = np.array(get_observation()["tcp_cartesian_pos"], dtype=float)
    err = float(np.linalg.norm(o[:3] - np.array(p, dtype=float)))
    print("target", p, "reached", o[:3], "err", round(err, 4), "grip", round(o[7], 3))
    return o, err

# 1) lower in front of the opening, outside the microwave footprint
o, err = go([x_t, float(tcp[1]), z_t])

# 2) slide into the opening along +y in steps
last_good = [x_t, float(tcp[1]), z_t]
for y in [0.04, 0.09, 0.13, 0.17]:
    p = [x_t, y, z_t]
    o, err = go(p)
    if err > 0.03:
        print("tracking error large -> backing off to", last_good)
        o, err = go(last_good)
        break
    last_good = p

CONTEXT["inserted_tcp"] = [float(v) for v in o[:3]]
try:
    mp, mq = get_object_pose("yellow and white mug")
    print("mug now", mp)
    if mp is not None:
        CONTEXT["mug_in_mw"] = [float(v) for v in mp]
except Exception as e:
    print("mug ERR", e)
print("final tcp", get_observation()["tcp_cartesian_pos"])
