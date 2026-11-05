try:
    p1 = get_object_pose("basket", use_multiview=False)
    print("single-view basket:", p1)
except Exception as e:
    print("single-view err:", e)

names = ["basket on the right", "brown woven basket", "basket with white lining",
         "gray wicker basket", "square basket"]
done = False
for n in names:
    try:
        res = transport("alphabet soup can", n)
        print("transport OK with", n, "->", res)
        done = True
        break
    except Exception as e:
        print("transport err with", n, ":", e)

obs = get_observation()
print("after tcp:", obs["tcp_cartesian_pos"])
print("joints:", obs["robot_joint_pos"])
