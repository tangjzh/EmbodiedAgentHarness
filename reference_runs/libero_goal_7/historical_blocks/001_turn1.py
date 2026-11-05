import numpy as np, math

def qmul(a,b):
    w1,x1,y1,z1=a; w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
def rotz(q,th):
    return qmul(np.array([math.cos(th/2),0,0,math.sin(th/2)]), q)

stove_c = np.array([0.409,0.205,0.018])
knob=None
for name in ["black knob on the back of the stove","stove control knob","knob"]:
    try:
        p,_q = get_object_pose(name)
        print(name,p)
        if p is not None and 0.18<p[0]<0.40 and 0.10<p[1]<0.32 and p[2]<0.15:
            knob=np.array(p); break
    except Exception as e:
        print(name,"ERR",e)
if knob is None:
    knob=np.array([stove_c[0]-0.11, stove_c[1]-0.01, 0.06])
    print("fallback knob",knob)
knob[2]=float(np.clip(knob[2],0.03,0.10))
print("knob used",knob)

obs=get_observation()
q0=np.array(obs["tcp_cartesian_pos"][3:7])
# canonical top-down
qd=np.array([0.0,1.0,0.0,0.0])
open_gripper()
move_to([knob[0],knob[1],knob[2]+0.15],qd)
move_to([knob[0],knob[1],knob[2]+0.0],qd,z_approach=0.08)
obs=get_observation(); print("at knob tcp",obs["tcp_cartesian_pos"])
close_gripper()
obs=get_observation(); print("after close",obs["tcp_cartesian_pos"][7])
pos=np.array(obs["tcp_cartesian_pos"][:3])
for k in range(1,5):
    q=rotz(qd, -math.radians(25*k))
    move_to(pos.tolist(), q.tolist())
obs=get_observation(); print("after rotate",obs["tcp_cartesian_pos"], obs["robot_joint_pos"])
open_gripper()
obs=get_observation()
p=np.array(obs["tcp_cartesian_pos"][:3]); qq=obs["tcp_cartesian_pos"][3:7]
move_to([p[0],p[1],p[2]+0.12],qq)
print("done",get_observation()["tcp_cartesian_pos"])
