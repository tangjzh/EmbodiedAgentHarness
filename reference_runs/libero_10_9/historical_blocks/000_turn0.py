import numpy as np, math

def quat_to_mat(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

open_gripper()
obs = get_observation()
print("tcp", obs["tcp_cartesian_pos"])

names = ["yellow and white mug", "yellow mug", "white yellow mug", "mug with yellow handle"]
target_name, mug_pose = None, None
for n in names:
    try:
        p = get_object_pose(n)
        if p is not None:
            target_name, mug_pose = n, p
            break
    except Exception as e:
        print("pose fail", n, e)
print("mug", target_name, mug_pose)

grey_pose = None
for n in ["grey-white mug", "grey mug", "white mug"]:
    try:
        grey_pose = get_object_pose(n)
        if grey_pose is not None:
            break
    except Exception as e:
        print("grey fail", n, e)
print("grey", grey_pose)

grasp = None
if target_name is not None:
    for attempt in range(3):
        try:
            g = sample_grasp_pose(target_name)
        except Exception as e:
            print("grasp fail", e); g = None
        print("grasp sample", g)
        if g is None:
            continue
        gp = np.array(g[0]); R = quat_to_mat(g[1])
        ok = True
        if mug_pose is not None:
            d = np.linalg.norm(gp[:2] - np.array(mug_pose[0])[:2])
            print("xy dist to mug", d)
            if d > 0.10: ok = False
        if R[2,2] > -0.5:  # tool z should point mostly downward
            print("approach not downward", R[:,2]); ok = False
        if grey_pose is not None and np.linalg.norm(gp[:2]-np.array(grey_pose[0])[:2]) < 0.06:
            ok = False
        if ok:
            grasp = (gp, np.array(g[1])); break

if grasp is not None:
    print("using sampled grasp", grasp)
    move_to(list(grasp[0]), list(grasp[1]), z_approach=0.12)
elif mug_pose is not None:
    c = np.array(mug_pose[0], dtype=float)
    q = np.array(obs["tcp_cartesian_pos"][3:7])
    side = 1.0
    if grey_pose is not None:
        side = 1.0 if (c[1] - grey_pose[0][1]) > 0 else -1.0
    # fingers close along base y with current downward orientation; straddle wall at +/- y
    tgt = c + np.array([0.0, side*0.04, 0.0])
    tgt[2] = c[2] + 0.02
    print("fallback target", tgt)
    move_to(list(tgt), list(q), z_approach=0.12)

obs = get_observation()
print("final tcp", obs["tcp_cartesian_pos"])
