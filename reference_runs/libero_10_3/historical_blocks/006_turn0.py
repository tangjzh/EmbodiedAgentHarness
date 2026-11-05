import numpy as np, math

obs = get_observation()
tcp = np.array(obs["tcp_cartesian_pos"][:3]); quat = list(obs["tcp_cartesian_pos"][3:7])
q_now = np.array(obs["robot_joint_pos"][:7])
print("tcp", tcp, "quat", quat, "grip", obs["tcp_cartesian_pos"][7])

GRIP = 0.0  # keep holding

def move_to(target, quat, n=10, steps_each=6):
    global q_now
    start = np.array(get_observation()["tcp_cartesian_pos"][:3])
    target = np.array(target)
    for i in range(1, n+1):
        p = start + (target - start) * i / n
        try:
            q = solve_ik(list(p), quat)
        except Exception as e:
            print("IK fail", p, e); return False
        if q is None:
            print("IK None", p); return False
        command_joints(list(q), GRIP, steps=steps_each)
    command_joints(list(q), GRIP, steps=15)
    return True

# Lift the bowl straight up
move_to([tcp[0], tcp[1], 0.28], quat, n=8, steps_each=6)
obs = get_observation()
print("after lift tcp", obs["tcp_cartesian_pos"])

for name in ["white_cabinet_1_bottom_region", "bottom drawer", "white cabinet", "black bowl"]:
    try:
        r = get_object_pose(name)
        print(name, "->", r)
    except Exception as e:
        print(name, "ERR", e)
