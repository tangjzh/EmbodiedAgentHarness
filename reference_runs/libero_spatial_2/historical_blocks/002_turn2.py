import numpy as np
obs = get_observation()
cur = np.array(obs["tcp_cartesian_pos"][:3]); q0 = list(obs["tcp_cartesian_pos"][3:7])
PLATE = np.array([0.712159499, 0.182088464, 0.000557777609])
bowl = np.array([0.65374016, 0.09019127, 0.03397964])
try:
    p, r = get_object_pose("black bowl left of the plate")
    print("est", p)
    if p is not None and p[1] < 0.15 and abs(p[0]-bowl[0]) < 0.08 and abs(p[1]-bowl[1]) < 0.08:
        bowl = np.array(p)
except Exception as e:
    print("err", e)
print("bowl", bowl)

if obs["tcp_cartesian_pos"][7] < 0.9:
    open_gripper()
# rim grasp on +y side (same strategy as initial successful grasp), top-down
gp = np.array([bowl[0] - 0.012, bowl[1] + 0.047, 0.03])
move_to([cur[0], cur[1], 0.22], q0)
move_to([gp[0], gp[1], 0.20], q0)
move_to([gp[0], gp[1], 0.10], q0)
move_to(list(gp), q0)
close_gripper()
obs = get_observation(); g1 = obs["tcp_cartesian_pos"][7]; print("after close", g1)
move_to([gp[0], gp[1], 0.12], q0)
move_to([gp[0], gp[1], 0.24], q0)
obs = get_observation(); g2 = obs["tcp_cartesian_pos"][7]; print("after lift", g2)
if g2 > 0.025:
    move_to([PLATE[0], PLATE[1], 0.24], q0)
    move_to([PLATE[0], PLATE[1], 0.18], q0)
obs = get_observation()
print("final tcp", obs["tcp_cartesian_pos"])
try:
    print("bowl now", get_object_pose("black bowl held by gripper"))
except Exception as e:
    print(e)
