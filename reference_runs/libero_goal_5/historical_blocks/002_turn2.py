import numpy as np

def est(name):
    for mv in (True, False):
        try:
            p = get_object_pose(name, use_multiview=mv)
            if p is not None and p[0] is not None:
                return np.array(p[0], dtype=float)
        except Exception as e:
            print(name, "pose fail mv=%s" % mv, e)
    return None

# 1) raise gripper away from the table so the next approach doesn't sweep the plate
try:
    lift("red-rimmed plate", height=0.15)
except Exception as e:
    print("lift FAIL", e)

target = np.array([0.71, 0.20])
p0 = est("red-rimmed plate")
if p0 is None:
    p0 = np.array([0.697, -0.034, 0.001])
print("plate before", np.round(p0, 3))

def sweep(fn, p):
    delta = target - p[:2]
    dist = float(np.linalg.norm(delta))
    d = delta / max(dist, 1e-6)
    dist = float(np.clip(dist, 0.03, 0.30))
    print(fn.__name__, "dir", np.round(d, 3), "dist", round(dist, 3))
    try:
        r = fn("red-rimmed plate", direction=(float(d[0]), float(d[1]), 0.0), distance=dist)
        print("result", r)
    except Exception as e:
        print(fn.__name__, "FAIL", e)
    try:
        lift("red-rimmed plate", height=0.12)
    except Exception as e:
        print("lift FAIL", e)

sweep(toggle_switch, p0)
p1 = est("red-rimmed plate")
print("plate after toggle", None if p1 is None else np.round(p1, 3))

if p1 is not None:
    moved = np.linalg.norm(p1[:2] - p0[:2])
    remaining = np.linalg.norm(target - p1[:2])
    print("moved", round(moved, 3), "remaining", round(remaining, 3))
    if remaining > 0.06:
        if moved > 0.03:
            sweep(toggle_switch, p1)
        else:
            sweep(open_drawer, p1)
        p2 = est("red-rimmed plate")
        print("plate final", None if p2 is None else np.round(p2, 3))

b = est("small blue box")
print("blue box", None if b is None else np.round(b, 3))
obs = get_observation()
print("tcp", np.round(np.array(obs["tcp_cartesian_pos"]), 3))
