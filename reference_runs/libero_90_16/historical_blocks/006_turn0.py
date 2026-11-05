import numpy as np, math

obs = get_observation()
tcp = np.array(obs["tcp_cartesian_pos"][:3]); quat = list(obs["tcp_cartesian_pos"][3:7])
grip_now = obs["tcp_cartesian_pos"][7]
print("tcp", tcp, "grip", grip_now)

poses = {}
for n in ["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"]:
    try:
        p = get_object_pose(n)
        poses[n] = np.array(p[0]) if p is not None else None
    except Exception as e:
        poses[n] = None
    print(n, poses[n])

# held bowl: elevated one / near tcp
held_c = None
table = []
for n, p in poses.items():
    if p is None: continue
    if p[2] > 0.06 and np.linalg.norm(p[:2]-tcp[:2]) < 0.15:
        held_c = p
    else:
        table.append((n, p))
print("held", held_c, "table", table)

# grasp offset tcp - bowl center
offset = np.array([0.758 - 0.700, 0.1795 - 0.195])
if held_c is not None:
    off2 = tcp[:2] - held_c[:2]
    if np.linalg.norm(off2) < 0.1:
        offset = off2
print("offset", offset)

# middle bowl: table bowl closest to (0.596, 0.21), not near held bowl/tcp
ref = np.array([0.596, 0.2097])
mid = ref.copy()
cands = [p for n, p in table if np.linalg.norm(p[:2] - (tcp[:2]-offset)) > 0.06 and p[2] < 0.06]
if cands:
    cands.sort(key=lambda p: np.linalg.norm(p[:2]-ref))
    if np.linalg.norm(cands[0][:2]-ref) < 0.08:
        mid = cands[0][:2]
print("middle bowl xy", mid)

target_xy = mid + offset
print("target tcp xy", target_xy)

def move_to(pos, grip, nseg=8, steps=10):
    o = get_observation()
    start = np.array(o["tcp_cartesian_pos"][:3])
    pos = np.array(pos)
    for i in range(1, nseg+1):
        wp = start + (pos-start)*i/nseg
        j = solve_ik(list(wp), quat)
        if j is None:
            print("IK fail at", wp); return False
        command_joints(list(j), grip, steps=steps)
    command_joints(list(j), grip, steps=10)
    o = get_observation()
    print("reached", np.round(o["tcp_cartesian_pos"][:3],4), "grip", round(o["tcp_cartesian_pos"][7],3))
    return True

g = grip_now  # keep held
move_to([target_xy[0], target_xy[1], 0.18], g, nseg=10, steps=10)
move_to([target_xy[0], target_xy[1], 0.12], g, nseg=6, steps=10)
move_to([target_xy[0], target_xy[1], 0.085], g, nseg=6, steps=12)
# release
o = get_observation()
j = solve_ik(list(o["tcp_cartesian_pos"][:3]), quat)
for k in range(4):
    command_joints(list(j), 1.0, steps=10)
o = get_observation(); print("after open grip", o["tcp_cartesian_pos"][7])
move_to([target_xy[0], target_xy[1], 0.20], 1.0, nseg=6, steps=10)
for n in ["akita_black_bowl_3"]:
    try: print(n, get_object_pose(n))
    except Exception as e: print(e)
