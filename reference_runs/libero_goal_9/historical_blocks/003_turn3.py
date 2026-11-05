import numpy as np
q0 = [0.3446891173014271, 0.5813528008267995, 0.660907468249136, -0.3262202504402737]
prior_b = np.array([0.46061433, -0.04737905, 0.13559997])
prior_off = np.array([0.46149868, -0.07155102, 0.17229177]) - prior_b  # tcp - bottle when it held

def tcp():
    return get_observation()["tcp_cartesian_pos"]

def est(name, ok):
    out = []
    for mv in [True, False]:
        try:
            r = get_object_pose(name, use_multiview=mv)
            if r is not None and r[0] is not None:
                p = np.array(r[0], dtype=float).reshape(-1)
                if p.size == 3 and ok(p):
                    out.append(p); print(name, mv, p); break
                print(name, mv, "rejected", p)
        except Exception as e:
            print(name, mv, "err", e)
    return out

# bottle estimate
bl = est("wine bottle", lambda p: np.linalg.norm(p[:2] - prior_b[:2]) < 0.15 and 0.02 < p[2] < 0.3)
b = bl[0] if bl else prior_b.copy()
print("bottle used", b)

held = False
attempts = [(-0.015, -0.03), (-0.025, -0.05), (0.0, -0.01)]  # (dy extra, dz) relative to prior offset
for i, (dy, dz) in enumerate(attempts):
    gpos = b + prior_off + np.array([0.0, dy, dz])
    gpos[2] = float(np.clip(gpos[2], 0.08, 0.2))
    print("attempt", i, gpos)
    open_gripper()
    c = tcp()
    if c[2] < 0.3:
        # back off along +tool approach direction then up
        move_to([c[0], c[1] + 0.12, c[2] + 0.1], q0)
    move_to(list(gpos), q0, z_approach=0.15)
    close_gripper()
    g = tcp()[7]; print("grip", g)
    if g > 0.05:
        ok = True
        for z in [gpos[2] + 0.03, gpos[2] + 0.08, gpos[2] + 0.18, 0.45, 0.62]:
            move_to([gpos[0], gpos[1], z], q0)
            gg = tcp()[7]; print("lift", z, gg)
            if gg < 0.03:
                ok = False; break
        if ok:
            held = True; break
        print("lost during lift")
    # re-perceive bottle for next attempt
    open_gripper()
    c = tcp()
    move_to([c[0], c[1] + 0.1, max(c[2], 0.35)], q0)
    bl = est("wine bottle", lambda p: np.linalg.norm(p[:2] - prior_b[:2]) < 0.2 and 0.02 < p[2] < 0.3)
    if bl: b = bl[0]

# rack estimate
cands = [np.array([0.59130703, -0.17965405, 0.06935176]), np.array([0.41121423, -0.18474572, 0.11734148])]
cands += est("wooden wine rack", lambda p: 0.1 < p[0] < 0.75 and -0.45 < p[1] < 0.0)
C = np.array(cands)
rack = np.median(C, axis=0)
print("rack used", rack)

if held:
    hz = 0.62
    c = tcp()
    move_to([c[0], c[1], hz], q0)
    move_to([(c[0] + rack[0]) / 2, (c[1] + rack[1]) / 2, hz], q0)
    move_to([rack[0], rack[1], hz], q0)
    o = tcp(); print("hover over rack", o)
else:
    print("could not secure bottle")
