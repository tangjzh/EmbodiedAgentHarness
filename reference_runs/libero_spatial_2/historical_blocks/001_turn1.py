import numpy as np

def qmat(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

obs = get_observation()
tcp = np.array(obs["tcp_cartesian_pos"][:3]); q0 = list(obs["tcp_cartesian_pos"][3:7])
grip = obs["tcp_cartesian_pos"][7]
print("start tcp", tcp, "grip", grip)
PLATE = np.array([0.712159499, 0.182088464, 0.000557777609])

if grip < 0.03:
    # bowl was dropped -> regrasp
    open_gripper()
    move_to([tcp[0], tcp[1], 0.25], q0)
    bowl = None
    for name in ["black bowl", "akita black bowl", "black bowl left of the plate"]:
        try:
            p, r = get_object_pose(name)
            print("est", name, p)
            if p is not None and p[1] < 0.15 and np.linalg.norm(np.array(p[:2]) - PLATE[:2]) > 0.06:
                bowl = np.array(p); break
        except Exception as e:
            print("err", e)
    print("bowl", bowl)
    grasp = None
    for i in range(3):
        try:
            g = sample_grasp_pose("black bowl")
        except Exception as e:
            print("gerr", e); g = None
        print("grasp cand", g)
        if g is None or g[0] is None:
            continue
        gp = np.array(g[0]); gq = list(g[1])
        zax = qmat(gq)[:, 2]
        ok = zax[2] < -0.7 and gp[1] < 0.15
        if bowl is not None:
            ok = ok and np.linalg.norm(gp[:2] - bowl[:2]) < 0.09
        if ok:
            grasp = (gp, gq); break
    if grasp is None and bowl is not None:
        # heuristic rim grasp like before (rim offset +y), top-down
        gp = np.array([bowl[0] - 0.015, bowl[1] + 0.048, max(bowl[2] + 0.005, 0.03)])
        grasp = (gp, q0)
        print("heuristic grasp", gp)
    if grasp is not None:
        gp, gq = grasp
        gp = gp.copy(); gp[2] = max(gp[2], 0.025)
        move_to([gp[0], gp[1], gp[2] + 0.15], gq)
        move_to(list(gp), gq)
        close_gripper()
        obs = get_observation(); print("after close grip", obs["tcp_cartesian_pos"][7])
        move_to([gp[0], gp[1], 0.25], gq)
        obs = get_observation(); grip = obs["tcp_cartesian_pos"][7]
        print("after lift grip", grip, obs["tcp_cartesian_pos"][:3])
        q0 = gq

obs = get_observation()
grip = obs["tcp_cartesian_pos"][7]
if grip > 0.03:
    cur = np.array(obs["tcp_cartesian_pos"][:3]); q = list(obs["tcp_cartesian_pos"][3:7])
    move_to([cur[0], cur[1], max(cur[2], 0.25)], q)
    move_to([PLATE[0], PLATE[1], 0.25], q)
    move_to([PLATE[0], PLATE[1], 0.18], q)
    obs = get_observation()
    print("final tcp", obs["tcp_cartesian_pos"])
    try:
        print("bowl now", get_object_pose("black bowl"))
    except Exception as e:
        print(e)
else:
    print("no grasp, grip", grip)
