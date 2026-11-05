import numpy as np
obs = get_observation()
tcp = np.array(obs["tcp_cartesian_pos"], dtype=float)
start_pos = tcp[:3].copy()
quat = tcp[3:7].copy()
print("start tcp", start_pos, "grip", tcp[7])

# Re-apply the grip (the mug walls stop the fingers) so the mug doesn't slip.
close_gripper()
obs = get_observation()
print("after close grip", obs["tcp_cartesian_pos"][7])

# Lift in two steps, keeping the orientation fixed.
for dz in [0.07, 0.15]:
    target = start_pos + np.array([0.0, 0.0, dz])
    move_to(target.tolist(), quat.tolist())
    o = get_observation()
    print("tcp now", o["tcp_cartesian_pos"][:3], "grip", o["tcp_cartesian_pos"][7])

try:
    p = get_object_pose("yellow and white mug")
    print("mug pose estimate", p)
except Exception as e:
    print("pose est failed", e)
