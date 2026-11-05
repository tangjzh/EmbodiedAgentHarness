import numpy as np, math

def _get(obs, key):
    if isinstance(obs, dict):
        if key in obs:
            return obs[key]
        for v in obs.values():
            if isinstance(v, dict) and key in v:
                return v[key]
    return None

def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def cur_joints():
    o = get_observation()
    return np.array(_get(o, 'robot_joint_pos'), dtype=float)[:7]

def move_joint_interp(qt, grip=1.0, maxstep=0.03, ticks=3, settle=0):
    qc = cur_joints()
    d = np.max(np.abs(qt - qc))
    n = max(int(math.ceil(d / maxstep)), 1)
    for i in range(1, n + 1):
        command_joints(list(qc + (qt - qc) * i / n), grip, steps=ticks)
    if settle > 0:
        command_joints(list(qt), grip, steps=settle)

def move_cart(target, quat, grip=1.0, seg=0.03, maxstep=0.03, settle=15):
    o = get_observation()
    start = np.array(_get(o, 'tcp_cartesian_pos'), dtype=float)[:3]
    target = np.array(target, dtype=float)
    dist = np.linalg.norm(target - start)
    n = max(int(math.ceil(dist / seg)), 1)
    for i in range(1, n + 1):
        p = start + (target - start) * i / n
        qt = np.array(solve_ik(list(p), list(quat)), dtype=float)[:7]
        move_joint_interp(qt, grip, maxstep=maxstep, ticks=3, settle=0)
    qt = np.array(solve_ik(list(target), list(quat)), dtype=float)[:7]
    command_joints(list(qt), grip, steps=settle)
    o = get_observation()
    t = np.array(_get(o, 'tcp_cartesian_pos'), dtype=float)
    print("target", np.round(target, 4), "reached", np.round(t[:3], 4), "grip", round(float(t[-1]), 3))
    return t

obs = get_observation()
av = obs.get('agentview') if isinstance(obs, dict) else None
if isinstance(av, dict):
    print("agentview keys", list(av.keys()))
    for k, v in av.items():
        try:
            print(" ", k, np.array(v).shape)
        except Exception:
            print(" ", k, type(v))
else:
    print("agentview type", type(av))

bottle = np.array([0.51401447, 0.03444402, 0.09860237])
board = np.array([0.56092635, -0.20943801, 0.01707453])

bc = None
for name in ['bowl', 'white bowl', 'grey bowl', 'metal bowl', 'ceramic bowl']:
    try:
        p, q = get_object_pose(name)
    except Exception as e:
        print("pose fail", name, e); continue
    if p is None:
        print(name, "None"); continue
    p = np.array(p, dtype=float)
    print(name, np.round(p, 4))
    if np.linalg.norm(p[:2] - bottle[:2]) > 0.07 and 0.5 < p[0] < 0.85 and -0.2 < p[1] < 0.1 and p[2] < 0.12:
        bc = p
        print("accepted bowl estimate from", name)
        break

if bc is None:
    bc = np.array([0.67, -0.045, 0.03])
    print("FALLBACK bowl estimate", bc)

# choose rim point with gripper footprint clearance
r = 0.065
def clearance(phi):
    rim = bc[:2] + r * np.array([math.cos(phi), math.sin(phi)])
    ax = np.array([math.cos(phi), math.sin(phi)])   # opening axis (radial)
    perp = np.array([-ax[1], ax[0]])
    c = 1e9
    for s in np.linspace(-0.1, 0.1, 11):
        for w in (-0.03, 0.0, 0.03):
            pt = rim + s * ax + w * perp
            c = min(c, np.linalg.norm(pt - bottle[:2]) - 0.04)   # bottle
            c = min(c, pt[1] - (-0.16))                           # wooden board side
            c = min(c, 0.07 - pt[1])                              # cabinet side
            c = min(c, 0.80 - pt[0])                              # reach limit
    return c

best_phi, best_c = 0.0, -1e9
for k in range(36):
    phi = 2 * math.pi * k / 36
    c = clearance(phi)
    if c > best_c:
        best_c, best_phi = c, phi
phi = best_phi
print("chosen phi deg", round(math.degrees(phi), 1), "clearance", round(best_c, 4))

theta = phi - math.pi / 2
while theta > math.pi / 2: theta -= math.pi
while theta < -math.pi / 2: theta += math.pi
qg = qmul(np.array([math.cos(theta / 2), 0, 0, math.sin(theta / 2)]), np.array([0.0, 1.0, 0.0, 0.0]))
rim = bc[:2] + r * np.array([math.cos(phi), math.sin(phi)])
z_fin = float(np.clip(bc[2] + 0.01, 0.035, 0.08))
print("rim xy", np.round(rim, 4), "z_fin", z_fin, "quat", np.round(qg, 4))

t = np.array(_get(get_observation(), 'tcp_cartesian_pos'), dtype=float)
move_cart([t[0], t[1], 0.32], qg, 1.0, seg=0.04)            # lift / rotate in place
move_cart([rim[0], rim[1], 0.32], qg, 1.0, seg=0.04)        # translate high
move_cart([rim[0], rim[1], z_fin + 0.08], qg, 1.0, seg=0.03)
tf = move_cart([rim[0], rim[1], z_fin], qg, 1.0, seg=0.015, maxstep=0.015, settle=25)
print("final joints", np.round(_get(get_observation(), 'robot_joint_pos'), 4))

try:
    CONTEXT['bowl_center'] = bc.tolist(); CONTEXT['rim_phi'] = phi
    CONTEXT['grasp_quat'] = qg.tolist(); CONTEXT['rim_r'] = r
    CONTEXT['rim_target'] = [float(rim[0]), float(rim[1]), z_fin]
except Exception as e:
    print("context store fail", e)
