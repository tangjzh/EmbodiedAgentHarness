import numpy as np, math

obs = get_observation()
tcp = np.array(obs["tcp_cartesian_pos"][:3]); q0 = list(obs["tcp_cartesian_pos"][3:7])
c = np.array([0.69152453, -0.03796758, 0.01731175])
y = np.array([-0.93999096, -0.3411943]); y /= np.linalg.norm(y)
r = 0.058
cands = [c[:2] + r*y, c[:2] - r*y]
rim = min(cands, key=lambda p: np.linalg.norm(p - tcp[:2]))
print("rim xy", rim)

def move_to(pos, quat, grip, total_steps=40, chunks=4):
    j_target = np.array(solve_ik(list(pos), list(quat)))
    j_now = np.array(get_observation()["robot_joint_pos"][:7])
    for i in range(1, chunks+1):
        j = j_now + (j_target - j_now) * i / chunks
        command_joints(list(j), grip, steps=max(1, total_steps//chunks))
    command_joints(list(j_target), grip, steps=10)
    o = get_observation()
    print("cmd", np.round(pos,4), "reached", np.round(o["tcp_cartesian_pos"][:3], 4), "grip", round(o["tcp_cartesian_pos"][7],3))
    return o

# Pre-grasp above the rim
move_to([rim[0], rim[1], 0.12], q0, 1.0, total_steps=40)
# Descend in stages, watching for contact (tracking error)
for z in [0.08, 0.055, 0.04, 0.03]:
    o = move_to([rim[0], rim[1], z], q0, 1.0, total_steps=30, chunks=3)
    err = o["tcp_cartesian_pos"][2] - z
    if err > 0.012:
        print("contact/stall detected at z", z, "err", err)
        # back off slightly
        cur = o["tcp_cartesian_pos"][:3]
        move_to([cur[0], cur[1], cur[2] + 0.005], q0, 1.0, total_steps=10, chunks=1)
        break

# Close gripper holding position
o = get_observation()
j_hold = list(o["robot_joint_pos"][:7])
for k in range(4):
    command_joints(j_hold, 0.0, steps=15)
o = get_observation()
print("after close tcp", np.round(o["tcp_cartesian_pos"], 4))
print("joints", np.round(o["robot_joint_pos"], 4))
