import json,numpy as np,time
from kin import fk, ik
from motion import topdown_R
o=json.load(open('data/runs/ep5_staged_obs.json')); q0=np.array(o['robot']['agent']['qpos'][0][:7])
t0=time.time()
for yaw_deg in [67.5, 90, 112.5]:
    for z in [0.08, 0.12, 0.16, 0.20, 0.25, 0.30]:
        row=[]
        for y in [-0.1, 0.0, 0.1, 0.2]:
            # bisection on x for reachability (warm-started IK)
            lo,hi=-0.2,0.35; q=q0.copy()
            qs,ok,_=ik(np.array([lo,y,z]),topdown_R(np.radians(yaw_deg)),q,iters=300)
            if not ok: row.append(None); continue
            qlo=qs
            for _ in range(9):
                mid=0.5*(lo+hi); qs,ok,_=ik(np.array([mid,y,z]),topdown_R(np.radians(yaw_deg)),qlo,iters=300)
                if ok: lo,qlo=mid,qs
                else: hi=mid
            row.append(round(lo,3))
        print('yaw %.1f z %.2f  xmax at y=-0.1,0,0.1,0.2:'%(yaw_deg,z), row)
print('time %.1f'%(time.time()-t0))
