import numpy as np, time
t0 = time.time()
obs = get_observation()
cur = np.array(obs["tcp_cartesian_pos"][:3]); q0 = list(obs["tcp_cartesian_pos"][3:7])
PLATE = np.array([0.712159499, 0.182088464, 0.000557777609])
OTHER = np.array([0.67, 0.313])

bowl = None
for name in ["black bowl", "black bowl near the center of the table", "metal bowl"]:
    try:
        p, r = get_object_pose(name)
        print("est", name, p)
        if p is not None and np.linalg.norm(np.array(p[:2]) - OTHER) > 0.09 \
           and np.linalg.norm(np.array(p[:2]) - PLATE[:2]) > 0.06 and 0.4 < p[0] < 0.85:
            bowl = np.array(p); break
    except Exception as e:
        print("err", e)
if bowl is None:
    bowl = np.array([0.654, 0.090, 0.03])
print("bowl", bowl)

if obs["tcp_cartesian_pos"][7] < 0.9:
    open_gripper()
move_to([cur[0], cur[1], 0.22], q0)

offsets = [(-0.012, 0.047), (0.0, 0.040), (0.0, -0.045), (0.045, 0.0)]
held = False
for dx, dy in offsets:
    if time.time() - t0 > 130:
        break
    gp = np.array([bowl[0] + dx, bowl[1] + dy, 0.032])
    move_to([gp[0], gp[1], 0.16], q0)
    move_to(list(gp), q0)
    close_gripper()
    g = get_observation()["tcp_cartesian_pos"][7]
    print("offset", dx, dy, "grip", g)
    if g > 0.025:
        move_to([gp[0], gp[1], 0.24], q0)
        g2 = get_observation()["tcp_cartesian_pos"][7]
        print("after lift", g2)
        if g2 > 0.025:
            held = True
            break
    open_gripper()
    move_to([gp[0], gp[1], 0.16], q0)

if held:
    move_to([PLATE[0], PLATE[1], 0.24], q0)
    move_to([PLATE[0], PLATE[1], 0.19], q0)
obs = get_observation()
print("held", held, "final tcp", obs["tcp_cartesian_pos"])
