import numpy as np, math

obs = get_observation()
tcp = np.array(obs["tcp_cartesian_pos"][:3]); quat = list(obs["tcp_cartesian_pos"][3:7])
g = float(obs["tcp_cartesian_pos"][7])
print("tcp", tcp, "grip", g)

# middle bowl from accumulated context (consistent estimates)
mid = np.array([0.5958, 0.2096])
# grasp offset tcp - bowl center (orientation unchanged since grasp)
offset = np.array([0.757955549577344 - 0.700, 0.1794708572938487 - 0.195])
target_xy = mid + offset
print("target tcp xy", target_xy)

def move_to(pos, grip, nseg=8, steps=10):
    o = get_observation()
    start = np.array(o["tcp_cartesian_pos"][:3])
    pos = np.array(pos, dtype=float)
    j = None
    for i in range(1, nseg+1):
        wp = start + (pos-start)*i/nseg
        jj = solve_ik(list(wp), quat)
        if jj is None:
            print("IK fail at", wp); break
        j = list(jj)
        command_joints(j, grip, steps=steps)
    if j is not None:
        command_joints(j, grip, steps=10)
    o = get_observation()
    print("reached", np.round(o["tcp_cartesian_pos"][:3],4), "grip", round(o["tcp_cartesian_pos"][7],3))
    return j

move_to([target_xy[0], target_xy[1], 0.175], g, nseg=8, steps=10)
move_to([target_xy[0], target_xy[1], 0.12], g, nseg=6, steps=10)
j = move_to([target_xy[0], target_xy[1], 0.085], g, nseg=6, steps=12)

# release
o = get_observation()
jr = solve_ik(list(o["tcp_cartesian_pos"][:3]), quat)
jr = list(jr) if jr is not None else j
for k in range(4):
    command_joints(jr, 1.0, steps=10)
o = get_observation(); print("after open grip", o["tcp_cartesian_pos"][7])
move_to([target_xy[0], target_xy[1], 0.20], 1.0, nseg=6, steps=10)

try:
    print("bowl_3", get_object_pose("akita_black_bowl_3"))
except Exception as e:
    print("pose err", e)
