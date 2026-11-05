import numpy as np, math

obs = get_observation()
tcp = np.array(obs["tcp_cartesian_pos"][:3]); q0 = np.array(obs["tcp_cartesian_pos"][3:7])
print("tcp", tcp, q0)

pose = None
for name in ["black_bowl", "akita_black_bowl_1", "black bowl", "bowl"]:
    try:
        p = get_object_pose(name)
        if p is not None:
            pose = (name, p); break
    except Exception as e:
        print("pose fail", name, e)
print("bowl pose", pose)

grasp = None
if pose is not None:
    try:
        grasp = sample_grasp_pose(pose[0])
    except Exception as e:
        print("grasp fail", e)
print("grasp", grasp)

def qmat(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
R = qmat(q0)
print("tcp axes x,y,z:", R[:,0], R[:,1], R[:,2])

def move_to(pos, quat, grip, total_steps=60, chunks=6):
    j_target = np.array(solve_ik(list(pos), list(quat)))
    j_now = np.array(get_observation()["robot_joint_pos"][:7])
    for i in range(1, chunks+1):
        j = j_now + (j_target - j_now) * i / chunks
        command_joints(list(j), grip, steps=max(1, total_steps//chunks))
    command_joints(list(j_target), grip, steps=10)
    o = get_observation()
    print("reached", np.round(o["tcp_cartesian_pos"], 4))
    return o

if pose is not None:
    c = np.array(pose[1][0])
    y = R[:,1].copy(); y[2] = 0; y /= np.linalg.norm(y)
    r = 0.065
    cands = [c[:2] + r*y, c[:2] - r*y]
    rim = min(cands, key=lambda p: np.linalg.norm(p - tcp[:2]))
    print("bowl center", c, "rim target xy", rim)
    pre = np.array([rim[0], rim[1], max(c[2] + 0.12, 0.12)])
    move_to(pre, q0, 1.0, total_steps=60)
