import numpy as np

def q2R(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

name = "yellow and white mug"
obs = get_observation()
tcp = np.array(obs["tcp_cartesian_pos"])
print("TCP now:", tcp)

mug = None
try:
    mug = get_object_pose(name)
    print("mug pose:", mug)
except Exception as e:
    print("pose fail", e)

grasp = None
for i in range(3):
    try:
        g = sample_grasp_pose(name)
    except Exception as e:
        print("grasp fail", e); g = None
    print("grasp sample", i, g)
    if g is None:
        continue
    gp = np.array(g[0]); gq = np.array(g[1])
    zax = q2R(gq)[:, 2]
    ok = zax[2] < -0.8
    if mug is not None:
        ok = ok and np.linalg.norm(gp[:2] - np.array(mug[0])[:2]) < 0.10
    ok = ok and gp[2] > 0.0
    print(" tool z axis", zax, "ok", ok)
    if ok:
        grasp = (gp, gq)
        break

if grasp is not None:
    gp, gq = grasp
    move_to(gp.tolist(), gq.tolist(), z_approach=0.08)
    close_gripper()
else:
    print("No acceptable grasp; not moving.")

obs = get_observation()
print("TCP after:", obs["tcp_cartesian_pos"])
print("joints:", obs["robot_joint_pos"])
