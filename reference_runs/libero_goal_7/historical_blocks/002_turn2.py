import numpy as np, math

def qmul(a,b):
    w1,x1,y1,z1=a; w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
def rotz(q,th):
    return qmul(np.array([math.cos(th/2),0,0,math.sin(th/2)]), np.array(q))

obs=get_observation()
tcp=np.array(obs["tcp_cartesian_pos"]); root=np.array(obs["robot_cartesian_pos"][:3])
cam=obs["robot0_eye_in_hand"]
rgb=depth=K=T=None
def scan(d,prefix=""):
    global rgb,depth,K,T
    if isinstance(d,dict):
        for k,v in d.items():
            kl=str(k).lower()
            if isinstance(v,dict): scan(v,prefix+kl+"."); continue
            a=np.asarray(v) if not isinstance(v,(str,bytes)) else None
            if a is None: continue
            print("cam key",prefix+kl,a.shape,a.dtype)
            if a.ndim==3 and a.shape[-1]==3 and rgb is None: rgb=a
            elif ("depth" in kl) and a.ndim>=2 and depth is None: depth=a.squeeze()
            elif a.shape==(3,3) and K is None: K=a
            elif a.shape==(4,4) and T is None: T=a
scan(cam)
heur=np.array([0.27,0.20,0.05])
knob=heur.copy(); top_z=None
try:
    img=rgb.astype(float)
    H,W=img.shape[:2]
    s=lambda r: int(r*H/256)
    mask=np.zeros((H,W),bool)
    sub=img[s(95):s(190), s(105):s(200)].max(axis=2)<45
    mask[s(95):s(190), s(105):s(200)]=sub
    vs,us=np.nonzero(mask)
    u,v=us.mean(),vs.mean()
    print("knob px",u*256/W,v*256/H,"n",len(us))
    d=np.median(depth[vs,us]); print("depth stats",np.median(depth),d, depth.min(), depth.max())
    fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
    cands=[]
    for flip in [False,True]:
        vv=(H-1-v) if flip else v
        x=(u-cx)*d/fx; y=(vv-cy)*d/fy
        for pc in [np.array([x,y,d,1]),np.array([x,-y,-d,1])]:
            for M in [T,np.linalg.inv(T)]:
                w=M@pc; ct=M[:3,3]
                off=root-ct if np.linalg.norm(ct-root)>0.25 else np.zeros(3)
                cands.append(w[:3]+off)
    cands=sorted(cands,key=lambda c: np.linalg.norm(c[:2]-heur[:2])+ (0 if -0.02<c[2]<0.15 else 1))
    print("cands",[np.round(c,3) for c in cands[:4]])
    c=cands[0]
    if np.linalg.norm(c[:2]-heur[:2])<0.08 and -0.02<c[2]<0.15:
        knob=c; top_z=c[2]
except Exception as e:
    print("deproj fail",e)
print("knob est",knob)
gz=float(np.clip((top_z if top_z is not None else 0.06)-0.015,0.035,0.08))
q_cur=tcp[3:7]
qg=rotz(q_cur, math.radians(90))   # fingers perpendicular to handle bar, lowers joint7
open_gripper()
move_to([knob[0],knob[1],gz+0.12],qg.tolist())
move_to([knob[0],knob[1],gz],qg.tolist())
close_gripper()
o=get_observation(); print("grip",o["tcp_cartesian_pos"],"j7",o["robot_joint_pos"][6])
pos=np.array(o["tcp_cartesian_pos"][:3])
if o["tcp_cartesian_pos"][7]>0.03:
    for k in range(1,4):
        move_to(pos.tolist(), rotz(qg, math.radians(30*k)).tolist())
    o=get_observation(); print("rotated",o["tcp_cartesian_pos"],"j7",o["robot_joint_pos"][6])
else:
    print("missed knob")
open_gripper()
o=get_observation(); p=o["tcp_cartesian_pos"]
move_to([p[0],p[1],p[2]+0.12],p[3:7])
print("end",get_observation()["tcp_cartesian_pos"])
