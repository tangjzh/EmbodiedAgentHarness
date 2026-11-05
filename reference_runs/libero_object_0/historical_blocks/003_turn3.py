def tcp():
    return get_observation()["tcp_cartesian_pos"]

print("start tcp:", tcp())
# Recover the dropped can
for step, fn in [("reach", lambda: reach("alphabet soup can")),
                 ("grasp", lambda: grasp("alphabet soup can")),
                 ("lift", lambda: lift("alphabet soup can", height=0.15))]:
    try:
        r = fn()
        print(step, "ok", r)
    except Exception as e:
        print(step, "err:", e)
    print(" tcp:", tcp())

# Find a basket name whose estimate is plausible (basket is on the image right, i.e. +Y side)
chosen = None
for n in ["wicker basket on the right side", "basket on the right"]:
    try:
        p = get_object_pose(n)
        print(n, "->", p)
        if p is not None and p[0] is not None and p[0][1] > -0.02:
            chosen = n
            break
    except Exception as e:
        print(n, "err:", e)
        break

if chosen is not None:
    try:
        r = transport("alphabet soup can", chosen)
        print("transport ok", r)
    except Exception as e:
        print("transport err:", e)
else:
    print("no plausible basket estimate; not transporting")

o = get_observation()
print("final tcp:", o["tcp_cartesian_pos"])
print("joints:", o["robot_joint_pos"])
