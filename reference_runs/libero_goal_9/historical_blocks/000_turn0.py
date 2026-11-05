import numpy as np
obs = get_observation()
tcp = np.array(obs["tcp_cartesian_pos"][:3]); q = list(obs["tcp_cartesian_pos"][3:7])
print("tcp", tcp, "q", q, "grip", obs["tcp_cartesian_pos"][7])
rack = None
for name in ["wooden wine rack", "wine rack", "rack"]:
    try:
        r = get_object_pose(name)
        print(name, r)
        if r is not None:
            rack = r; break
    except Exception as e:
        print("err", name, e)
cab = None
for name in ["dark cabinet", "cabinet"]:
    try:
        c = get_object_pose(name)
        print(name, c)
        if c is not None:
            cab = c; break
    except Exception as e:
        print("err", name, e)
try:
    print("bottle", get_object_pose("wine bottle"))
except Exception as e:
    print("bottle err", e)

safe_z = 0.55
if cab is not None:
    safe_z = max(safe_z, float(cab[0][2]) + 0.45)
safe_z = min(safe_z, 0.7)
# lift vertically first
move_to([tcp[0], tcp[1], safe_z], q)
o = get_observation(); print("after lift", o["tcp_cartesian_pos"])
if rack is not None:
    rp = np.array(rack[0])
    move_to([rp[0], rp[1], safe_z], q)
    o = get_observation(); print("over rack", o["tcp_cartesian_pos"])
