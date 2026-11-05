import numpy as np
obs = get_observation()
tcp = np.array(obs["tcp_cartesian_pos"][:3]); q = list(obs["tcp_cartesian_pos"][3:7])
print("tcp", tcp, "grip", obs["tcp_cartesian_pos"][7])

plate = None
for name in ["plate", "red-rimmed plate", "plate_1"]:
    try:
        p = get_object_pose(name)
        if p is not None:
            plate = p; print("plate via", name, p); break
    except Exception as e:
        print("fail", name, e)

other = None
for name in ["akita_black_bowl_2", "black bowl", "akita black bowl"]:
    try:
        o = get_object_pose(name)
        print("bowl est", name, o)
    except Exception as e:
        print("fail", name, e)

if plate is not None:
    px, py, pz = plate[0][:3]
    safe_z = max(tcp[2] + 0.08, pz + 0.22)
    # lift first
    move_to([tcp[0], tcp[1], safe_z], q)
    # move across at height
    move_to([px, py, safe_z], q)
    # descend to hover above plate (still well above surface)
    hover_z = max(pz + 0.17, 0.15)
    move_to([px, py, hover_z], q)
    obs = get_observation()
    print("final tcp", obs["tcp_cartesian_pos"])
