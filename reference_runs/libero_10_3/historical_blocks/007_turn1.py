import numpy as np, math
obs = get_observation()
quat = list(obs["tcp_cartesian_pos"][3:7])
GRIP = 0.0
def move_to(target, quat, n=10, steps_each=6):
    start = np.array(get_observation()["tcp_cartesian_pos"][:3])
    target = np.array(target)
    q = None
    for i in range(1, n+1):
        p = start + (target - start) * i / n
        q = solve_ik(list(p), quat)
        if q is None:
            print("IK None", p); return False
        command_joints(list(q), GRIP, steps=steps_each)
    command_joints(list(q), GRIP, steps=15)
    return True

move_to([0.62, -0.17, 0.30], quat, n=10, steps_each=6)
obs = get_observation()
print("tcp", obs["tcp_cartesian_pos"])
for name in ["open bottom drawer of white cabinet", "drawer", "white_cabinet_1_bottom_region", "black bowl"]:
    try:
        print(name, "->", get_object_pose(name))
    except Exception as e:
        print(name, "ERR", e)
