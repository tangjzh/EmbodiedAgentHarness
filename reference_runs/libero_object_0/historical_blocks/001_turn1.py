names = ["basket", "wicker basket", "white basket", "storage basket"]
chosen = None
for n in names:
    try:
        p = get_object_pose(n)
        print(n, "->", p)
        if p is not None and p[0] is not None:
            chosen = n
            break
    except Exception as e:
        print(n, "err:", e)

if chosen is not None:
    try:
        res = transport("alphabet soup can", chosen)
        print("transport result:", res)
    except Exception as e:
        print("transport err:", e)
else:
    print("no basket name perceived")

obs = get_observation()
print("after tcp:", obs["tcp_cartesian_pos"])
print("joints:", obs["robot_joint_pos"])
