import numpy as np, math

def find_key(d, k):
    if isinstance(d, dict):
        if k in d: return d[k]
        for v in d.values():
            r = find_key(v, k)
            if r is not None: return r
    return None

obs = get_observation()
tcp = np.array(find_key(obs, 'tcp_cartesian_pos'), dtype=float)
q_now = np.array(find_key(obs, 'robot_joint_pos'), dtype=float)[:7]
print("TCP", tcp)

poses = {}
for n in ["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"]:
    try:
        p = get_object_pose(n)
        if p is not None:
            poses[n] = (np.array(p[0], float), np.array(p[1], float))
            print(n, poses[n][0])
    except Exception as e:
        print("fail", n, e)
if len(poses) < 2:
    for n in ["black_bowl_1", "black_bowl_2", "black_bowl_3", "black bowl at the front"]:
        try:
            p = get_object_pose(n)
            if p is not None:
                poses[n] = (np.array(p[0], float), np.array(p[1], float))
                print(n, poses[n][0])
        except Exception as e:
            print("fail", n, e)
plate = None
for n in ["plate_1", "plate"]:
    try:
        p = get_object_pose(n)
        if p is not None:
            plate = np.array(p[0], float); print("plate", plate); break
    except Exception as e:
        print("fail", n, e)

CONTEXT['bowl_poses'] = {k: v[0].tolist() for k, v in poses.items()}

def move_to(pos, quat, ticks=60, grip=1.0, chunks=6):
    global q_now
    qt = np.array(solve_ik(list(pos), list(quat)), float)[:7]
    for i in range(1, chunks + 1):
        qi = q_now + (qt - q_now) * i / chunks
        command_joints(qi.tolist(), grip, steps=max(1, ticks // chunks))
    command_joints(qt.tolist(), grip, steps=15)
    o = get_observation()
    q_now = np.array(find_key(o, 'robot_joint_pos'), float)[:7]
    t = np.array(find_key(o, 'tcp_cartesian_pos'), float)
    print("reached", t[:3], "target", pos)
    return t

if poses:
    # front bowl = nearest to robot base (smallest x)
    name = min(poses, key=lambda k: poses[k][0][0])
    c = poses[name][0]
    CONTEXT['front_bowl'] = name
    print("front bowl:", name, c)
    others = [v[0] for k, v in poses.items() if k != name]
    if plate is not None: others.append(plate)
    r = 0.06
    q0 = np.array([0.0, 1.0, 0.0, 0.0])
    grasp = None
    try:
        g = sample_grasp_pose(name)
        if g is not None:
            gp = np.array(g[0], float); gq = np.array(g[1], float)
            print("sampled grasp", gp, gq)
            dxy = np.linalg.norm(gp[:2] - c[:2])
            # require near rim and roughly top-down
            if 0.02 < dxy < 0.10 and abs(gq[0]) < 0.4 and abs(gq[3]) < 0.95:
                grasp = (gp, gq)
    except Exception as e:
        print("grasp fail", e)
    if grasp is None:
        # rim point along +/- y (default finger axis along y => radial straddle)
        best = None
        for s in [1, -1]:
            rp = c + np.array([0, s * r, 0])
            dmin = min([np.linalg.norm(rp[:2] - o[:2]) for o in others] + [9])
            if best is None or dmin > best[0]: best = (dmin, rp)
        rp = best[1]
        gp = np.array([rp[0], rp[1], c[2] + 0.01]); gq = q0
        grasp = (gp, gq)
    gp, gq = grasp
    print("using grasp", gp, gq)
    CONTEXT['grasp'] = (gp.tolist(), gq.tolist())
    # lift up and over, then descend
    move_to([tcp[0], tcp[1], max(tcp[2], gp[2] + 0.15)], gq, ticks=40)
    move_to(gp + np.array([0, 0, 0.12]), gq, ticks=80)
    move_to(gp + np.array([0, 0, 0.05]), gq, ticks=40)
    move_to(gp, gq, ticks=40)
else:
    print("no bowl poses found")
