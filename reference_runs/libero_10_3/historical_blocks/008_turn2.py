import numpy as np, math

obs = get_observation()
quat = list(obs["tcp_cartesian_pos"][3:7])
GRIP = 0.0

def move_to(target, quat, n=10, steps_each=6):
    start = np.array(get_observation()["tcp_cartesian_pos"][:3])
    target = np.array(target, dtype=float)
    q = None
    for i in range(1, n+1):
        p = start + (target - start) * i / n
        q = solve_ik(list(p), quat)
        if q is None:
            print("IK None", p); return False
        command_joints(list(q), GRIP, steps=steps_each)
    command_joints(list(q), GRIP, steps=20)
    return True

# ---------- collect arrays from observation ----------
arrs = {}
def walk(d, pre=""):
    if isinstance(d, dict):
        for k, v in d.items():
            walk(v, pre + "/" + str(k))
    elif isinstance(d, (list, tuple)) and len(d) > 0 and isinstance(d[0], dict):
        for i, v in enumerate(d):
            walk(v, pre + "/" + str(i))
    else:
        try:
            a = np.asarray(d, dtype=float)
            arrs[pre] = a
        except Exception:
            pass
walk(obs)
for k, a in arrs.items():
    if a.size > 1:
        print(k, a.shape, (float(np.nanmin(a)), float(np.nanmax(a))) if a.size < 10**7 else "")

Ks = {k: a for k, a in arrs.items() if a.shape == (3, 3)}
Ts = {k: a for k, a in arrs.items() if a.shape == (4, 4)}
imgs = {k: a for k, a in arrs.items() if a.ndim == 3 and a.shape[2] == 3 and a.shape[0] > 50}
depths = {k: (a[..., 0] if a.ndim == 3 else a) for k, a in arrs.items()
          if (a.ndim == 2 and a.shape[0] > 50) or (a.ndim == 3 and a.shape[2] == 1 and a.shape[0] > 50)}

tcp_now = np.array(obs["tcp_cartesian_pos"][:3])
bowl_pt = tcp_now + np.array([0, 0, -0.03])
exp_px = np.array([65.0, 95.0])  # bowl in 256x256 agentview (u,v)
flipGL = np.diag([1, -1, -1, 1.0])

best = None
for kn, K in Ks.items():
    for tn, T in Ts.items():
        for inv in (False, True):
            for gl in (False, True):
                try:
                    W2C = np.linalg.inv(T) if not inv else T
                    if gl:
                        W2C = flipGL @ W2C
                    pc = W2C @ np.append(bowl_pt, 1.0)
                    if pc[2] <= 0.05: continue
                    uv = (K @ pc[:3])[:2] / pc[2]
                    W = 2 * K[0, 2]; H = 2 * K[1, 2]
                    for vflip in (False, True):
                        u, v = uv
                        if vflip: v = H - v
                        uvn = np.array([u * 256 / W, v * 256 / H])
                        err = np.linalg.norm(uvn - exp_px)
                        if best is None or err < best[0]:
                            best = (err, kn, tn, inv, gl, vflip, K, W2C, W, H)
                except Exception as e:
                    pass

target_xy = np.array([0.60, 0.10])  # fallback from image-geometry estimate
if best is not None:
    err, kn, tn, inv, gl, vflip, K, W2C, W, H = best
    print("best calib", err, kn, tn, inv, gl, vflip)
    if err < 25:
        C2W = np.linalg.inv(W2C)
        cam_c = C2W[:3, 3]
        cands = []
        for (pu, pv) in [(182, 142), (180, 150), (185, 138)]:
            u = pu * W / 256; v = pv * H / 256
            if vflip: v = H - v
            ray_c = np.linalg.inv(K) @ np.array([u, v, 1.0])
            ray_w = C2W[:3, :3] @ ray_c
            for zp in (0.08, 0.11):
                t = (zp - cam_c[2]) / ray_w[2]
                pt = cam_c + t * ray_w
                cands.append(pt)
                print("ray-plane", (pu, pv), zp, pt)
        cands = np.array(cands)
        med = np.median(cands, axis=0)
        if -0.1 < med[1] < 0.4 and 0.35 < med[0] < 0.85:
            target_xy = med[:2]
print("target_xy", target_xy)

# carry high, then over the drawer, then descend
move_to([tcp_now[0], tcp_now[1], 0.36], quat, n=4, steps_each=6)
move_to([target_xy[0], target_xy[1], 0.36], quat, n=14, steps_each=6)
move_to([target_xy[0], target_xy[1], 0.21], quat, n=8, steps_each=6)
obs = get_observation()
print("final tcp", obs["tcp_cartesian_pos"])
