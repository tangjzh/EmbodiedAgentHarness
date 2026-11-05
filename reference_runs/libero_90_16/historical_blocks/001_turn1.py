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
Q0 = [0.0, 1.0, 0.0, 0.0]

def move_to(pos, quat, ticks=60, grip=1.0, chunks=6):
    global q_now
    qt = np.array(solve_ik(list(map(float, pos)), list(map(float, quat))), float)[:7]
    for i in range(1, chunks + 1):
        qi = q_now + (qt - q_now) * i / chunks
        command_joints(qi.tolist(), grip, steps=max(1, ticks // chunks))
    command_joints(qt.tolist(), grip, steps=15)
    o = get_observation()
    q_now = np.array(find_key(o, 'robot_joint_pos'), float)[:7]
    t = np.array(find_key(o, 'tcp_cartesian_pos'), float)
    print("reached", np.round(t[:3], 4), "target", np.round(np.array(pos, float), 4))
    return t

# 1) lift straight up away from the middle bowl
move_to([tcp[0], tcp[1], 0.20], Q0, ticks=40)
move_to([0.50, 0.10, 0.30], Q0, ticks=50)   # get arm out of camera view somewhat

# 2) explore observation structure
obs = get_observation()
def walk(d, pre=""):
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            out += walk(v, pre + "/" + str(k))
    elif isinstance(d, (list, tuple)) and len(d) > 0 and not isinstance(d[0], (int, float)):
        for i, v in enumerate(d):
            out += walk(v, pre + "/" + str(i))
    else:
        a = np.asarray(d) if not isinstance(d, str) else None
        out.append((pre, a))
    return out
leaves = walk(obs)
for p, a in leaves:
    if a is not None and a.dtype != object:
        print(p, a.shape, a.dtype, (float(np.nanmin(a)), float(np.nanmax(a))) if a.size else None)
    else:
        print(p, type(a))

# 3) depth-based bowl localization
pts_all = []
leafd = {p: a for p, a in leaves}
depth_keys = [p for p, a in leaves if a is not None and a.dtype != object and 'depth' in p.lower() and a.ndim >= 2]
print("depth keys", depth_keys)
def find_near(prefix, words, shape):
    cands = [p for p, a in leaves if a is not None and a.dtype != object and a.shape == shape]
    best = None
    for p in cands:
        if any(w in p.lower() for w in words):
            # share longest common prefix with depth key
            cp = len(os.path.commonprefix([p, prefix])) if False else sum(1 for x, y in zip(p, prefix) if x == y)
            if best is None or cp > best[0]: best = (cp, p)
    return best[1] if best else None

centers_found = None
try:
    clouds = []
    for dk in depth_keys:
        D = np.squeeze(leafd[dk]).astype(float)
        if D.ndim != 2: continue
        K = find_near(dk, ['intr', 'k', 'matrix'], (3, 3))
        E = find_near(dk, ['extr', 'pose', 'cam', 'world', 'T'], (4, 4))
        print(dk, "K", K, "E", E)
        if K is None or E is None: continue
        Km = leafd[K].astype(float); Em = leafd[E].astype(float)
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
        print(dk, "plane score", best[0], best[2], best[3])
        if best[0] > 0.15: clouds.append(best[1])
    if clouds:
        P = np.concatenate(clouds, 0)
        sel = P[(P[:, 2] > 0.03) & (P[:, 2] < 0.09) & (P[:, 0] > 0.3) & (P[:, 0] < 0.95) & (np.abs(P[:, 1]) < 0.5)]
        print("candidate pts", len(sel))
        # connected components on 1 cm grid
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
            ctr = cp[:, :2].mean(0); ext = cp[:, :2].max(0) - cp[:, :2].min(0)
            rad = np.median(np.linalg.norm(cp[:, :2] - ctr, axis=1))
            print("cluster n", len(cp), "ctr", np.round(ctr, 3), "ext", np.round(ext, 3), "rad", round(rad, 3), "ztop", round(np.percentile(cp[:, 2], 95), 3))
            if 0.06 < max(ext) < 0.22: cs.append((ctr, rad, np.percentile(cp[:, 2], 95)))
        centers_found = cs
except Exception as e:
    import traceback; traceback.print_exc()

# 4) single-view perception cross-check
sv = {}
for n in ["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"]:
    try:
        p = get_object_pose(n, use_multiview=False)
        if p is not None: sv[n] = np.array(p[0], float); print("single-view", n, np.round(sv[n], 3))
    except Exception as e:
        print("sv fail", n, e)

plate = np.array([0.6495, 0.0166])
target = None
if centers_found and len(centers_found) >= 2:
    cs = sorted(centers_found, key=lambda t: t[0][0])
    CONTEXT['bowl_centers_depth'] = [[float(c[0][0]), float(c[0][1]), float(c[2])] for c in cs]
    target = cs[-1]   # largest x = front (nearest camera viewpoint)
    others = [c[0] for c in cs[:-1]] + [plate]
else:
    print("depth localisation failed; not moving to bowl")

if target is not None:
    c, rad, ztop = target
    r = rad if 0.04 < rad < 0.085 else 0.065
    best = None
    for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        rp = c + r * np.array([math.cos(ang), math.sin(ang)])
        dmin = min(np.linalg.norm(rp - o) for o in others)
        if best is None or dmin > best[0]: best = (dmin, ang, rp)
    _, ang, rp = best
    # finger axis default along base y for Q0; rotate yaw so finger axis is radial
    yaw = ang - math.pi / 2
    qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    def qmul(a, b):
        w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
        return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
    gq = qmul(qz, np.array(Q0))
    if gq[1] < 0: gq = -gq
    gz = max(ztop - 0.025, 0.025)
    gp = np.array([rp[0], rp[1], gz])
    print("front bowl ctr", np.round(c, 3), "r", round(r, 3), "grasp", np.round(gp, 3), np.round(gq, 3))
    CONTEXT['front_bowl_center'] = [float(c[0]), float(c[1]), float(ztop)]
    CONTEXT['grasp'] = (gp.tolist(), gq.tolist())
    move_to(gp + [0, 0, 0.15], gq, ticks=70)
    move_to(gp + [0, 0, 0.06], gq, ticks=40)
    move_to(gp, gq, ticks=40)
