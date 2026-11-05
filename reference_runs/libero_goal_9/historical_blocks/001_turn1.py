import numpy as np
obs = get_observation()
q = [0.3446891173014271, 0.5813528008267995, 0.660907468249136, -0.3262202504402737]
print("tcp now", obs["tcp_cartesian_pos"])
bp = get_object_pose("wine bottle")
print("bottle", bp)
b = np.array(bp[0])
open_gripper()
# go above bottle first (high, clear)
cur = np.array(get_observation()["tcp_cartesian_pos"][:3])
move_to([cur[0], cur[1], 0.45], q)
move_to([b[0], b[1], 0.45], q)
gz = float(np.clip(b[2] - 0.01, 0.08, 0.2))
move_to([b[0], b[1], gz], q, z_approach=0.12)
o = get_observation(); print("at grasp", o["tcp_cartesian_pos"])
close_gripper()
o = get_observation(); g = o["tcp_cartesian_pos"][7]; print("grip after close", g)
if g > 0.05:
    for z in [gz + 0.08, gz + 0.2, 0.35, 0.5]:
        move_to([b[0], b[1], z], q)
        o = get_observation(); print("lift", z, "grip", o["tcp_cartesian_pos"][7])
        if o["tcp_cartesian_pos"][7] < 0.03:
            print("LOST GRASP"); break
for name in ["wooden wine rack", "wine rack"]:
    try:
        print(name, get_object_pose(name))
    except Exception as e:
        print("err", e)
print("bottle after", get_object_pose("wine bottle"))
