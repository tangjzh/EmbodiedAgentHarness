import numpy as np, math

def _get(obs, key):
    if isinstance(obs, dict):
        if key in obs:
            return obs[key]
        for v in obs.values():
            if isinstance(v, dict) and key in v:
                return v[key]
    return None

obs = get_observation()
if isinstance(obs, dict):
    print("obs keys:", list(obs.keys()))
tcp = np.array(_get(obs, 'tcp_cartesian_pos'), dtype=float)
print("tcp", np.round(tcp, 4))

poses = {}
for name in ['black bowl', 'wine bottle', 'cabinet', 'wooden board']:
    try:
        p = get_object_pose(name)
        poses[name] = p
        print(name, p)
    except Exception as e:
        print("pose fail", name, e)

g = None
try:
    g = sample_grasp_pose('black bowl')
    print("grasp sample", g)
except Exception as e:
    print("grasp fail", e)

def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def move_to(pos, quat, grip=1.0, maxstep=0.03, ticks=3, settle=20):
    qt = np.array(solve_ik(list(pos), list(quat)), dtype=float)[:7]
    o = get_observation()
    qc = np.array(_get(o, 'robot_joint_pos'), dtype=float)[:7]
    d = np.max(np.abs(qt - qc))
    n = max(int(math.ceil(d / maxstep)), 1)
    for i in range(1, n + 1):
        command_joints(list(qc + (qt - qc) * i / n), grip, steps=ticks)
    command_joints(list(qt), grip, steps=settle)
    o = get_observation()
    t = np.array(_get(o, 'tcp_cartesian_pos'), dtype=float)
    print("target", np.round(pos, 4), "reached tcp", np.round(t, 4))
    return o, t

bc = np.array(poses['black bowl'][0], dtype=float)
obstacles = []
for k in ['wine bottle', 'cabinet', 'wooden board']:
    if k in poses and poses[k] is not None:
        obstacles.append(np.array(poses[k][0], dtype=float)[:2])

r = 0.065
best, best_s = 0.0, -1e9
for k in range(24):
    phi = 2 * math.pi * k / 24
    pt = bc[:2] + r * np.array([math.cos(phi), math.sin(phi)])
    s = min([np.linalg.norm(pt - ob) for ob in obstacles]) if obstacles else 1.0
    s -= 0.2 * max(0.0, pt[0] - 0.65)
    if s > best_s:
        best_s, best = s, phi
phi = best
theta = phi - math.pi / 2
while theta > math.pi / 2: theta -= math.pi
while theta < -math.pi / 2: theta += math.pi
q0 = np.array([0.0, 1.0, 0.0, 0.0])
qz = np.array([math.cos(theta / 2), 0, 0, math.sin(theta / 2)])
qg = qmul(qz, q0)
rim = bc[:2] + r * np.array([math.cos(phi), math.sin(phi)])

# rim height choice
z_rim = bc[2] + 0.03
if g is not None:
    try:
        gp = np.array(g[0], dtype=float)
        dxy = np.linalg.norm(gp[:2] - bc[:2])
        print("grasp dist from center", dxy)
        if 0.03 < dxy < 0.1 and bc[2] - 0.01 < gp[2] < bc[2] + 0.08:
            z_rim = gp[2]
    except Exception as e:
        print("grasp parse fail", e)
print("bowl center", np.round(bc, 4), "phi deg", math.degrees(phi), "rim", np.round(rim, 4), "z_rim", z_rim, "quat", np.round(qg, 4))

pre = [rim[0], rim[1], z_rim + 0.15]
o, t = move_to(pre, qg, 1.0)
mid = [rim[0], rim[1], z_rim + 0.06]
o, t = move_to(mid, qg, 1.0, maxstep=0.02)
fin = [rim[0], rim[1], z_rim]
o, t = move_to(fin, qg, 1.0, maxstep=0.015, settle=25)
print("final joints", np.round(_get(o, 'robot_joint_pos'), 4))

try:
    CONTEXT['bowl_center'] = bc.tolist(); CONTEXT['rim_phi'] = phi
    CONTEXT['grasp_quat'] = qg.tolist(); CONTEXT['rim_r'] = r
    CONTEXT['rim_target'] = list(fin)
except Exception as e:
    print("context store fail", e)
