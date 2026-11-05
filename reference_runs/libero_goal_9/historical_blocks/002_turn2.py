import numpy as np
q0 = [0.3446891173014271, 0.5813528008267995, 0.660907468249136, -0.3262202504402737]
prior_tcp = np.array([0.46149868, -0.07155102, 0.17229177])
prior_b = np.array([0.46061433, -0.04737905, 0.13559997])

def tcp():
    return get_observation()["tcp_cartesian_pos"]

b = None
try:
    bp = get_object_pose("wine bottle")
    if bp is not None and bp[0] is not None:
        b = np.array(bp[0], dtype=float)
except Exception as e:
    print("bottle err", e)
print("bottle est", b)
if b is None or np.linalg.norm(b[:2] - prior_b[:2]) > 0.15:
    b = prior_b.copy()

gpos, gq = None, None
try:
    gp = sample_grasp_pose("wine bottle")
    print("grasp sample", gp)
    if gp is not None and gp[0] is not None:
        p = np.array(gp[0], dtype=float)
        if np.linalg.norm(p[:2] - b[:2]) < 0.08 and 0.08 < p[2] < 0.25:
            gpos, gq = p, list(gp[1])
except Exception as e:
    print("grasp err", e)
if gpos is None:
    d = b - prior_b
    gpos = np.array([prior_tcp[0] + d[0], prior_tcp[1] + d[1], 0.16])
    gq = q0
print("using grasp", gpos, gq)

open_gripper()
c = tcp()
move_to([c[0], c[1], 0.45], q0)
move_to([gpos[0], gpos[1], 0.45], gq)
move_to(list(gpos), gq, z_approach=0.12)
print("at grasp", tcp())
close_gripper()
g = tcp()[7]; print("grip after close", g)

held = g > 0.05
if held:
    for z in [gpos[2] + 0.04, gpos[2] + 0.1, gpos[2] + 0.2, 0.42, 0.55]:
        move_to([gpos[0], gpos[1], z], gq)
        gg = tcp()[7]; print("lift", z, "grip", gg)
        if gg < 0.03:
            held = False; print("LOST GRASP"); break

rack = np.array([0.59130703, -0.17965405, 0.06935176])
try:
    rp = get_object_pose("wooden wine rack")
    print("rack est", rp)
    if rp is not None and rp[0] is not None:
        r = np.array(rp[0], dtype=float)
        if np.linalg.norm(r[:2] - rack[:2]) < 0.15:
            rack = r
except Exception as e:
    print("rack err", e)

if held:
    hz = 0.55
    move_to([(gpos[0] + rack[0]) / 2, (gpos[1] + rack[1]) / 2, hz], gq)
    print("mid grip", tcp()[7])
    move_to([rack[0], rack[1], hz], gq)
    o = tcp(); print("over rack", o)
    try:
        print("bottle now", get_object_pose("wine bottle"))
    except Exception as e:
        print("err", e)
