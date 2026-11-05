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

p0 = est("red-rimmed plate")
if p0 is None:
    p0 = np.array([0.697, -0.034, 0.001])
print("plate before", np.round(p0, 3))

target = np.array([0.70, 0.21])
delta = target - p0[:2]
dist = float(np.linalg.norm(delta))
d = delta / max(dist, 1e-6)
# extra travel to compensate for the 7cm standoff / rim engagement losses
dist = float(np.clip(dist + 0.05, 0.05, 0.32))
print("push dir", np.round(d, 3), "dist", round(dist, 3))
try:
    r = push("red-rimmed plate", direction=(float(d[0]), float(d[1]), 0.0), distance=dist)
    print("push result", r)
except Exception as e:
    print("push FAIL", e)

p1 = est("red-rimmed plate")
print("plate after", None if p1 is None else np.round(p1, 3))
obs = get_observation()
print("tcp", np.round(np.array(obs["tcp_cartesian_pos"]), 3))
