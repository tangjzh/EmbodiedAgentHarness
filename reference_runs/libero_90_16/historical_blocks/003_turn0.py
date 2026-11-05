import numpy as np

obs = get_observation()
tcp = np.array(obs["tcp_cartesian_pos"][:3])
q_cur = np.array(obs["robot_joint_pos"][:7])
print("start tcp", tcp, "joints", q_cur)

g_pos = np.array(CONTEXT["grasp"][0], dtype=float)
g_quat = np.array(CONTEXT["grasp"][1], dtype=float)

def move_to(p, quat, grip, steps):
    global q_cur
    q = np.array(solve_ik(list(p), list(quat)), dtype=float)
    if np.max(np.abs(q - q_cur)) > 1.0:
        print("Large IK jump", q - q_cur)
    command_joints(list(q), grip, steps=steps)
    q_cur = q
    return q

# Descend vertically (with small xy correction) in waypoints
start = tcp.copy()
waypoints = []
pre = np.array([g_pos[0], g_pos[1], 0.10])
for a in np.linspace(0, 1, 5)[1:]:
    waypoints.append(start + a * (pre - start))
for a in np.linspace(0, 1, 5)[1:]:
    waypoints.append(pre + a * (g_pos - pre))

for w in waypoints:
    move_to(w, g_quat, 1.0, 12)
# settle
command_joints(list(q_cur), 1.0, steps=15)
obs = get_observation()
print("at grasp tcp", obs["tcp_cartesian_pos"])

# Close gripper
command_joints(list(q_cur), 0.0, steps=40)
obs = get_observation()
print("after close tcp", obs["tcp_cartesian_pos"], "joints", obs["robot_joint_pos"])
