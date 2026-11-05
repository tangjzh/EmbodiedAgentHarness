obs = get_observation()
print("before tcp:", obs["tcp_cartesian_pos"])
try:
    basket = get_object_pose("woven basket")
    print("basket pose:", basket)
except Exception as e:
    print("basket pose err:", e)
res = transport("alphabet soup can", "woven basket")
print("transport result:", res)
obs = get_observation()
print("after tcp:", obs["tcp_cartesian_pos"])
print("joints:", obs["robot_joint_pos"])
