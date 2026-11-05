import numpy as np, math

def find_key(d, k):
    if isinstance(d, dict):
        if k in d: return d[k]
        for v in d.values():
            r = find_key(v, k)
            if r is not None: return r
    return None

obs = get_observation()
q_now = np.array(find_key(obs, 'robot_joint_pos'), float)[:7]
tcp = np.array(find_key(obs, 'tcp_cartesian_pos'), float)
Q0 = np.array([0.0, 1.0, 0.0, 0.0])

def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def move_to(pos, quat, ticks=60, grip=1.0, chunks=6):
    global q_now
    qt = np.array(solve_ik([float(v) for v in pos], [float(v) for v in quat]), float)[:7]
    for i in range(1, chunks + 1):
        qi = q_now + (qt - q_now) * i / chunks
        command_joints(qi.tolist(), grip, steps=max(1, ticks // chunks))
    command_joints(qt.tolist(), grip, steps=15)
    o = get_observation()
    q_now = np.array(find_key(o, 'robot_joint_pos'), float)[:7]
    t = np.array(find_key(o, 'tcp_cartesian_pos'), float)
    print("reached", np.round(t[:3], 4), "target", np.round(np.array(pos, float), 4))
    return t

# 1) lift straight up off the middle bowl, then move out of camera view
move_to([tcp[0], tcp[1], 0.20], Q0, ticks=40)
move_to([0.40, -0.12, 0.40], Q0, ticks=60)

# 2) observation leaves
obs = get_observation()
leaves = []
def walk(d, pre=""):
    if isinstance(d, dict):
        for k, v in d.items(): walk(v, pre + "/" + str(k))
        return
    if isinstance(d, str): return
    try:
        a = np.asarray(d)
        if a.dtype != object and a.size > 0 and np.issubdtype(a.dtype, np.number):
            leaves.append((pre, a)); return
    except Exception:
        pass
    if isinstance(d, (list, tuple)):
        for i, v in enumerate(d): walk(v, pre + "/" + str(i))
walk(obs)
for p, a in leaves:
    print(p, a.shape, a.dtype)
leafd = dict(leaves)

def sim(p, q):
    n = 0
    for x, y in zip(p, q):
        if x != y: break
        n += 1
    return n

def find_near(ref, words, shapes):
    best = None
    for p, a in leaves:
        if a.shape in shapes and any(w in p.lower() for w in words):
            s = sim(p, ref)
            if best is None or s > best[0]: best = (s, p)
    return best[1] if best else None

centers_found = None
try:
    clouds = []
    dks = [p for p, a in leaves if 'depth' in p.lower() and np.squeeze(a).ndim == 2]
    print("depth keys", dks)
    for dk in dks:
        D = np.squeeze(leafd[dk]).astype(float)
        if np.nanmax(D) > 50: D = D / 1000.0
        K = find_near(dk, ['intr', 'k', 'matrix'], [(3, 3)])
        E = find_near(dk, ['extr', 'pose', 'cam', 'world', 't'], [(4, 4), (3, 4)])
        print(dk, "K", K, "E", E)
        if K is None or E is None: continue
        Km = leafd[K].astype(float)
        Em = leafd[E].astype(float)
        if Em.shape == (3, 4): Em = np.vstack([Em, [0, 0, 0, 1]])
        H, W = D.shape
        vv, uu = np.mgrid[0:H, 0:W]
        z = D.ravel(); u = uu.ravel(); v = vv.ravel()
        ok = np.isfinite(z) & (z > 0.05) & (z < 5)
        z, u, v = z[ok], u[ok], v[ok]
        x = (u - Km[0, 2]) * z / Km[0, 0]; y = (v - Km[1, 2]) * z / Km[1, 1]
        Pc = np.stack([x, y, z, np.ones_like(z)], 0)
        best = None
        for inv in [False, True]:
            T = np.linalg.inv(Em) if inv else Em
            for flip in [False, True]:
                P = Pc.copy()
                if flip: P[1] *= -1; P[2] *= -1
                Pw = (T @ P)[:3].T
                score = np.mean((np.abs(Pw[:, 2]) < 0.015) & (Pw[:, 0] > 0.2) & (Pw[:, 0] < 1.0))
                if best is None or score > best[0]: best = (score, Pw, inv, flip)
        print(dk, "plane score", round(best[0], 3), best[2], best[3])
        if best[0] > 0.15: clouds.append(best[1])
    if clouds:
        P = np.concatenate(clouds, 0)
        sel = P[(P[:, 2] > 0.025) & (P[:, 2] < 0.09) & (P[:, 0] > 0.35) & (P[:, 0] < 0.95) & (np.abs(P[:, 1]) < 0.5)]
        print("candidate pts", len(sel))
        if len(sel) > 200000: sel = sel[::4]
        g = np.floor(sel[:, :2] / 0.01).astype(int)
        cells = {}
        for i, c in enumerate(map(tuple, g)): cells.setdefault(c, []).append(i)
        seen = set(); comps = []
        for c in cells:
            if c in seen: continue
            stack = [c]; seen.add(c); comp = []
            while stack:
                a = stack.pop(); comp += cells[a]
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        b = (a[0] + dx, a[1] + dy)
                        if b in cells and b not in seen: seen.add(b); stack.append(b)
            if len(comp) > 80: comps.append(sel[comp])
        cs = []
        for cp in comps:
            ctr = (cp[:, :2].max(0) + cp[:, :2].min(0)) / 2
            ext = cp[:, :2].max(0) - cp[:, :2].min(0)
            rad = float(np.percentile(np.linalg.norm(cp[:, :2] - ctr, axis=1), 90))
            ztop = float(np.percentile(cp[:, 2], 95))
            print("cluster n", len(cp), "ctr", np.round(ctr, 3), "ext", np.round(ext, 3), "rad", round(rad, 3), "ztop", round(ztop, 3))
            if 0.07 < max(ext) < 0.2 and min(ext) > 0.05: cs.append((ctr, rad, ztop))
        centers_found = cs
except Exception as e:
    print("depth error:", repr(e))

# 3) choose target (fallback from image geometry if depth unusable)
plate = np.array([0.6495, 0.0166])
middle_guess = np.array([0.596, 0.209])
fallback = (np.array([0.70, 0.195]), 0.065, 0.055)
target = None
others = None
if centers_found and 2 <= len(centers_found) <= 4:
    cs = sorted(centers_found, key=lambda t: t[0][0])
    CONTEXT['bowl_centers_depth'] = [[float(c[0][0]), float(c[0][1]), float(c[2])] for c in cs]
    target = cs[-1]   # largest x = front (nearest the camera viewpoint)
    others = [c[0] for c in cs[:-1]] + [plate]
    print("depth-based front bowl", np.round(target[0], 3))
else:
    print("using fallback front-bowl estimate")
    target = fallback
    others = [middle_guess, np.array([0.55, 0.07]), plate]

c, rad, ztop = target
r = rad - 0.005 if 0.045 < rad < 0.085 else 0.065
best = None
for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
    rp = c + r * np.array([math.cos(ang), math.sin(ang)])
    dmin = min(np.linalg.norm(rp - o) for o in others)
    reach_pen = max(0.0, rp[0] - 0.78) * 5
    sc = dmin - reach_pen
    if best is None or sc > best[0]: best = (sc, ang, rp)
_, ang, rp = best
# Q0 finger axis is along base y; rotate yaw so the finger axis is radial to the bowl
yaw = ang - math.pi / 2
yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2   # gripper symmetric under 180 deg
qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
gq = qmul(qz, Q0)
gz = float(np.clip(ztop - 0.02, 0.03, 0.06))
gp = np.array([rp[0], rp[1], gz])
print("front bowl ctr", np.round(c, 3), "r", round(r, 3), "grasp", np.round(gp, 3), np.round(gq, 3))
CONTEXT['front_bowl_center'] = [float(c[0]), float(c[1]), float(ztop)]
CONTEXT['grasp'] = (gp.tolist(), gq.tolist())

move_to(gp + np.array([0, 0, 0.16]), gq, ticks=70)
move_to(gp + np.array([0, 0, 0.06]), gq, ticks=40)
move_to(gp, gq, ticks=40)
