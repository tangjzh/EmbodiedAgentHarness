import numpy as np, math

def _get(obs, key):
    if isinstance(obs, dict):
        if key in obs: return obs[key]
        for v in obs.values():
            if isinstance(v, dict) and key in v: return v[key]
    return getattr(obs, key, None)

obs = get_observation()
try:
    print("obs keys:", list(obs.keys()))
except Exception as e:
    print("obs type", type(obs))
tcp = np.array(_get(obs, 'tcp_cartesian_pos'), dtype=float)
jp = np.array(_get(obs, 'robot_joint_pos'), dtype=float)
print("tcp", tcp, "joints", jp)

poses = {}
for name in ['black bowl', 'wine bottle', 'cabinet', 'wooden board']:
    try:
        p = get_object_pose(name)
        poses[name] = p
        print(name, p)
    except Exception as e:
        print("pose fail", name, e)

try:
    g = sample_grasp_pose('black bowl')
    print("grasp sample", g)
except Exception as e:
    g = None
    print("grasp fail", e)

def qmul(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def move_to(pos, quat, grip=1.0, maxstep=0.04, ticks=3, settle=15):
    qt = np.array(solve_ik(list(pos), list(quat)), dtype=float)[:7]
    o = get_observation()
    qc = np.array(_get(o, 'robot_joint_pos'), dtype=float)[:7]
    d = np.max(np.abs(qt - qc))
    n = max(int(math.ceil(d / maxstep)), 1)
    for i in range(1, n + 1):
        command_joints(list(qc + (qt - qc) * i / n), grip, steps=ticks)
    command_joints(list(qt), grip, steps=settle)
    o = get_observation()
    print("reached tcp", np.round(_get(o, 'tcp_cartesian_pos'), 4))
    return o

bowl = poses.get('black bowl')
bc = np.array(bowl[0], dtype=float)
obstacles = []
for k in ['wine bottle', 'cabinet', 'wooden board']:
    if k in poses and poses[k] is not None:
        obstacles.append(np.array(poses[k][0], dtype=float)[:2])

r = 0.06
best, best_s = None, -1
for k in range(16):
    phi = 2 * math.pi * k / 16
    pt = bc[:2] + r * np.array([math.cos(phi), math.sin(phi)])
    s = min([np.linalg.norm(pt - o) for o in obstacles]) if obstacles else 1.0
    # mild preference for staying not too far from robot
    s -= 0.1 * max(0, pt[0] - 0.65)
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
print("bowl center", bc, "phi(deg)", math.degrees(phi), "rim xy", rim, "quat", qg)

pre = [rim[0], rim[1], bc[2] + 0.15]
o = move_to(pre, qg, 1.0)
try:
    CONTEXT['bowl_center'] = bc.tolist(); CONTEXT['rim_phi'] = phi
    CONTEXT['grasp_quat'] = qg.tolist(); CONTEXT['rim_r'] = r
except Exception as e:
    print("context store fail", e)
