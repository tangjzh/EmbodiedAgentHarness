# Code block 0
import numpy as np
# Attempt 1: pick the BBQ sauce bottle (small brown/orange bottle, left-middle) with a fresh ctx.
c={'object':'bbq_sauce','container':'basket'}
r=None;pc=c;s={}
try:
    r=execute_option('pick',c);pc=r['data']['context'];s=get_abstract_state(pc)['abstract_state']
    print('pick',r['status'],r.get('termination_reason'),s.get('lying'),s.get('grasp_source'),s.get('failure_mode'),s.get('reach_err'),s.get('object_center'))
except Exception as e:
    print('pick exc',e)
g=float(get_observation()['robot_cartesian_pos'][7]);print('gap',g)
if r is not None and (r['status']=='command_completed' or (r['status']=='unknown' and g>0.04)):
    try:
        r2=execute_option('place',pc);p2=r2['data']['context'];s2=get_abstract_state(p2)['abstract_state']
        print('place',r2['status'],r2.get('termination_reason'),s2.get('release_z'),s2.get('placed_in_container'),s2.get('placement_reason'))
        if r2['status']=='unknown':
            v=execute_option('verify',p2);vs=get_abstract_state(v['data']['context'])['abstract_state']
            print('verify',vs.get('placed_in_container'),vs.get('placement_reason'))
    except Exception as e:
        print('place exc',e)

# Code block 1
import numpy as np
# The place's final check (outside_xy) likely measured the wrong bottle: Molmo's 'bbq_sauce' point (554,449) maps to the centre red bottle,
# while the wrist view shows a brown bottle lying inside the basket. Re-verify motion-free with a fresh ctx and a more specific name.
res={}
for name in ['brown_bbq_sauce_bottle_lying_in_basket','brown_bbq_sauce_bottle']:
    try:
        v=execute_option('verify',{'object':name,'container':'basket'})
        vs=get_abstract_state(v['data']['context'])['abstract_state']
        res[name]=(vs.get('placed_in_container'),vs.get('placement_reason'))
        print('verify',name,v['status'],vs.get('placed_in_container'),vs.get('placement_reason'),vs.get('object_center'))
        if vs.get('placed_in_container') is True:
            break
    except Exception as e:
        print('verify exc',name,e)
try:
    bp,_=get_object_pose('basket')
    op,_=get_object_pose('brown_bbq_sauce_bottle')
    print('basket',None if bp is None else np.round(bp,3),'bottle',None if op is None else np.round(op,3))
    if bp is not None and op is not None:
        print('xy dist',float(np.linalg.norm(np.array(bp[:2])-np.array(op[:2]))))
except Exception as e:
    print('pose exc',e)

# Code block 2
import numpy as np
# Main camera shows the bbq sauce (orange-red bottle, white cap) upright on the floor left of the basket; basket looks empty.
# Fresh pick with a specific visual name; container geometry is re-measured (not copied).
nc={'object':'orange_red_bbq_sauce_bottle_with_white_cap','container':'basket'}
r=None;pc=nc;s={}
try:
    r=execute_option('pick',nc);pc=r['data']['context'];s=get_abstract_state(pc)['abstract_state']
    print('pick',r['status'],r.get('termination_reason'),s.get('lying'),s.get('grasp_source'),s.get('failure_mode'),s.get('reach_ok'),s.get('object_center'))
except Exception as e:
    print('pick exc',e)
g=float(get_observation()['robot_cartesian_pos'][7]);print('gap',g)
if r is not None and r['status']=='error' and ('GraspNet' in str(r.get('termination_reason')) or 'candidates' in str(r.get('termination_reason'))):
    try:
        nc2={'object':'orange_red_bbq_sauce_bottle_with_white_cap','container':'basket','sampler_failed':True}
        r=execute_option('pick',nc2);pc=r['data']['context'];s=get_abstract_state(pc)['abstract_state']
        print('repick',r['status'],r.get('termination_reason'),s.get('failure_mode'))
        g=float(get_observation()['robot_cartesian_pos'][7]);print('gap',g)
    except Exception as e:
        print('repick exc',e)
if r is not None and (r['status']=='command_completed' or (r['status']=='unknown' and g>0.04)):
    try:
        r2=execute_option('place',pc);p2=r2['data']['context'];s2=get_abstract_state(p2)['abstract_state']
        print('place',r2['status'],r2.get('termination_reason'),s2.get('release_z'),s2.get('placed_in_container'),s2.get('placement_reason'))
        if r2['status']=='unknown':
            v=execute_option('verify',p2);vs=get_abstract_state(v['data']['context'])['abstract_state']
            print('verify',vs.get('placed_in_container'),vs.get('placement_reason'))
    except Exception as e:
        print('place exc',e)

# Code block 3
import numpy as np
# Previous pick: sampled side grasp closed empty (hand ended ~13 cm from object, loc_err 0.12 -> inconsistent multiview),
# geo retry was unreachable. Try once more with prefer='geo' (grasp_source was sampled first), fresh ctx, no copied container.
name='orange_red_bbq_sauce_bottle_with_white_cap'
nc={'object':name,'container':'basket','prefer':'geo'}
r=None;pc=nc;s={};placed=None
try:
    r=execute_option('pick',nc);pc=r['data']['context'];s=get_abstract_state(pc)['abstract_state']
    print('pick',r['status'],s.get('lying'),s.get('grasp_source'),s.get('failure_mode'),s.get('reach_ok'),s.get('object_center'))
except Exception as e:
    print('pick exc',e)
g=float(get_observation()['robot_cartesian_pos'][7]);print('gap',g)
if r is not None and (r['status']=='command_completed' or (r['status']=='unknown' and g>0.04)):
    try:
        r2=execute_option('place',pc);p2=r2['data']['context'];s2=get_abstract_state(p2)['abstract_state']
        placed=s2.get('placed_in_container')
        print('place',r2['status'],s2.get('release_z'),placed,s2.get('placement_reason'))
        if r2['status']=='unknown':
            v=execute_option('verify',p2);vs=get_abstract_state(v['data']['context'])['abstract_state']
            placed=vs.get('placed_in_container');print('verify',placed,vs.get('placement_reason'))
    except Exception as e:
        print('place exc',e)
else:
    # Bounded manual fallback: single-view (agentview) cloud, top-down grasp around the upper body.
    try:
        d=get_object_3d_points_and_masks_from_language(name,use_multiview=False)
        P=np.asarray(d['points_3d']);P=P[np.isfinite(P).all(1)]
        print('npts',len(P),'score',d.get('agentview_score'))
        ctr=np.median(P,axis=0);top=float(np.percentile(P[:,2],98))
        print('obj ctr',np.round(ctr,3),'top',round(top,3))
        db=get_object_3d_points_and_masks_from_language('wicker_basket',use_multiview=False)
        B=np.asarray(db['points_3d']);B=B[np.isfinite(B).all(1)]
        bc=np.median(B,axis=0);brim=float(np.percentile(B[:,2],97))
        print('basket ctr',np.round(bc,3),'rim',round(brim,3))
        q=np.array([0.0,1.0,0.0,0.0])
        cur=get_observation()['robot_cartesian_pos']
        # choose yaw so fingers close across the short horizontal axis
        obb=get_oriented_bounding_box_from_3d_points(P)
        R=np.asarray(obb['R']);ext=np.asarray(obb['extent'])
        hor=[i for i in range(3) if abs(R[2,i])<0.7]
        if len(hor)==2:
            short=hor[0] if ext[hor[0]]<ext[hor[1]] else hor[1]
            ax=R[:,short];yaw=np.arctan2(ax[1],ax[0])
            # top-down base quat (0,1,0,0) has fingers along y; rotate about world z
            a=yaw-np.pi/2
            qz=np.array([np.cos(a/2),0,0,np.sin(a/2)])
            w1,x1,y1,z1=qz;w2,x2,y2,z2=q
            q=np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])
            q=q/np.linalg.norm(q)
        gz=max(top-0.045,0.03)
        open_gripper()
        goto_pose(np.array([ctr[0],ctr[1],gz]),q,z_approach=0.15)
        close_gripper()
        goto_pose(np.array([ctr[0],ctr[1],gz+0.25]),q)
        g=float(get_observation()['robot_cartesian_pos'][7]);print('gap after lift',g)
        if g>0.04:
            hang=gz-float(np.min(P[:,2]))
            rz=max(brim+hang+0.03,gz+0.0)
            goto_pose(np.array([bc[0],bc[1],rz+0.12]),q)
            goto_pose(np.array([bc[0],bc[1],rz]),q)
            open_gripper()
            goto_pose(np.array([bc[0],bc[1],rz+0.15]),q)
        goto_home_joint_position()
        v=execute_option('verify',{'object':name,'container':'basket'})
        vs=get_abstract_state(v['data']['context'])['abstract_state']
        print('verify',vs.get('placed_in_container'),vs.get('placement_reason'),vs.get('object_center'))
    except Exception as e:
        print('manual exc',e)
        try: goto_home_joint_position()
        except Exception as e2: print('home exc',e2)

# Code block 4
import numpy as np
# Perception budget is exhausted, so options cannot run. Fall back to raw RGB-D from get_observation.
# The BBQ bottle (orange-red body, white cap) now lies on the floor just left of the basket.
# Plan: colour-segment it, back-project with depth, then do a top-down grasp across its short axis.
# Basket estimate from the earlier agentview cloud: ctr (0.666, 0.244), rim z ~0.111.
BC=np.array([0.666,0.244]);RIM=0.111

def backproject(obs,mask):
    a=obs['agentview'];D=np.asarray(a['images']['depth'],dtype=np.float64)
    K=np.asarray(a['intrinsics']);T=np.asarray(a['pose_mat'])
    vs,us=np.nonzero(mask);z=D[vs,us]
    xc=(us-K[0,2])*z/K[0,0];yc=(vs-K[1,2])*z/K[1,1]
    out={}
    for conv,pc in [('cv',np.stack([xc,yc,z],1)),('gl',np.stack([xc,-yc,-z],1))]:
        out[conv]=(T[:3,:3]@pc.T).T+T[:3,3]
    return out,D

def pick_conv(obs):
    # The correct convention puts the floor (bottom-centre strip of the image) near z=0.
    H,W=obs['agentview']['images']['depth'].shape
    m=np.zeros((H,W),bool);m[H-60:H-20,W//2-40:W//2+40]=True
    out,D=backproject(obs,m)
    best=None
    for k,P in out.items():
        P=P[np.isfinite(P).all(1)]
        if len(P)==0: continue
        zm=float(np.median(P[:,2]));print('conv',k,'floor z',round(zm,3),'x',round(float(np.median(P[:,0])),3))
        if best is None or abs(zm)<abs(best[1]): best=(k,zm)
    return best

def orange_points(obs,conv):
    rgb=np.asarray(obs['agentview']['images']['rgb']).astype(int)
    R,G,B=rgb[...,0],rgb[...,1],rgb[...,2]
    m=(R>140)&(G>40)&(G<150)&(B<90)&(R-G>50)
    out,_=backproject(obs,m);P=out[conv]
    P=P[np.isfinite(P).all(1)]
    P=P[(P[:,2]>-0.02)&(P[:,2]<0.25)&(P[:,0]>0.2)&(P[:,0]<0.9)&(np.abs(P[:,1])<0.5)]
    return P

try:
    obs=get_observation();best=pick_conv(obs);print('best',best)
    conv=best[0];depth_ok=abs(best[1])<0.05
    P=orange_points(obs,conv) if depth_ok else np.zeros((0,3))
    # exclude anything already inside the basket footprint
    if len(P): P=P[np.linalg.norm(P[:,:2]-BC,axis=1)>0.09]
    print('orange pts',len(P))
    if len(P)>=30:
        ctr=np.median(P,axis=0)
        P=P[np.linalg.norm(P[:,:2]-ctr[:2],axis=1)<0.12];ctr=np.median(P,axis=0)
        X=P[:,:2]-P[:,:2].mean(0);w,V=np.linalg.eigh(X.T@X/len(X))
        longax=V[:,1];short=V[:,0]
        print('ctr',np.round(ctr,3),'zmax',round(float(np.percentile(P[:,2],95)),3),'long',np.round(longax,2),'spread',np.round(np.sqrt(w),3))
    else:
        # fallback estimate from the earlier cloud of the lying bottle
        ctr=np.array([0.527,0.099,0.03]);short=np.array([-0.64,0.77]);print('fallback ctr')
    # Close across the short horizontal axis: base quat (0,1,0,0) has fingers along world y.
    yaw=np.arctan2(short[1],short[0]);a=yaw-np.pi/2
    a=(a+np.pi/2)%np.pi-np.pi/2  # keep the wrist rotation small
    qz=np.array([np.cos(a/2),0,0,np.sin(a/2)]);q0=np.array([0.,1.,0.,0.])
    w1,x1,y1,z1=qz;w2,x2,y2,z2=q0
    q=np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2]);q/=np.linalg.norm(q)
    gz=0.018  # lower than the failed 0.03 attempt, near the lying bottle's equator
    open_gripper()
    goto_pose(np.array([ctr[0],ctr[1],gz]),q,z_approach=0.15)
    close_gripper()
    goto_pose(np.array([ctr[0],ctr[1],0.25]),q)
    g=float(get_observation()['robot_cartesian_pos'][7]);print('gap after lift',g)
    if g>0.02:
        goto_pose(np.array([BC[0],BC[1],0.28]),q)
        goto_pose(np.array([BC[0],BC[1],RIM+0.05]),q)
        open_gripper()
        goto_pose(np.array([BC[0],BC[1],0.30]),q)
    else:
        open_gripper();goto_pose(np.array([ctr[0],ctr[1],0.25]),q)
    goto_home_joint_position()
    obs=get_observation()
    if depth_ok:
        P2=orange_points(obs,conv)
        inb=P2[np.linalg.norm(P2[:,:2]-BC,axis=1)<0.09] if len(P2) else P2
        outb=P2[np.linalg.norm(P2[:,:2]-BC,axis=1)>=0.09] if len(P2) else P2
        print('final orange pts in basket',len(inb),'outside',len(outb),
              'out ctr',None if len(outb)==0 else np.round(np.median(outb,axis=0),3))
except Exception as e:
    print('manual exc',e)
    try: goto_home_joint_position()
    except Exception as e2: print('home exc',e2)

# Code block 5
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')

# Code block 6
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')

# Code block 7
import numpy as np
# The bbq bottle (orange body, grey cap) lies on the floor just left of the basket. The basket looks empty.
# Perception options are budget-exhausted, so use raw RGB-D: colour-segment the bottle in an image ROI,
# back-project it with depth, pick the convention that puts the floor flat, then grasp top-down across the short axis.
def quat_mul(a,b):
    w1,x1,y1,z1=a;w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])

def bp(obs,mask,conv):
    a=obs['agentview'];D=np.asarray(a['images']['depth'],dtype=np.float64)
    if D.ndim==3: D=D[...,0]
    K=np.asarray(a['intrinsics']);T=np.asarray(a['pose_mat'])
    v,u=np.nonzero(mask);z=D[v,u]
    x=(u-K[0,2])*z/K[0,0];y=(v-K[1,2])*z/K[1,1]
    pc=np.stack([x,y,z],1) if conv=='cv' else np.stack([x,-y,-z],1)
    P=(T[:3,:3]@pc.T).T+T[:3,3]
    return P[np.isfinite(P).all(1)]

try:
    obs=get_observation()
    rgb=np.asarray(obs['agentview']['images']['rgb']).astype(int);H,W=rgb.shape[:2]
    D=np.asarray(obs['agentview']['images']['depth']);print('img',H,W,'depth range',float(np.nanmin(D)),float(np.nanmax(D)))
    sx,sy=W/800.0,H/512.0
    # Floor patch between the can and the basket, used to choose the convention.
    fm=np.zeros((H,W),bool);fm[int(380*sy):int(420*sy),int(470*sx):int(540*sx)]=True
    best=None
    for conv in ['cv','gl']:
        F=bp(obs,fm,conv)
        if len(F)==0: continue
        flat=float(np.std(F[:,2]));print(conv,'floor z',round(float(np.median(F[:,2])),3),'std',round(flat,4))
        if best is None or flat<best[1]: best=(conv,flat,float(np.median(F[:,2])))
    conv,_,floorz=best
    R,G,B=rgb[...,0],rgb[...,1],rgb[...,2]
    roi=np.zeros((H,W),bool);roi[int(225*sy):int(300*sy),int(410*sx):int(500*sx)]=True
    m=roi&(R>120)&(R-B>60)&(R-G>30)
    P=bp(obs,m,conv);print('bottle px',int(m.sum()),'pts',len(P))
    # Basket from the ROI using its white liner and wicker texture.
    broi=np.zeros((H,W),bool);broi[int(215*sy):int(320*sy),int(500*sx):int(615*sx)]=True
    Bp=bp(obs,broi,conv)
    bc=np.median(Bp,axis=0);rim=float(np.percentile(Bp[:,2],97))
    print('basket ctr',np.round(bc,3),'rim',round(rim,3))
    if len(P)>=20:
        ctr=np.median(P,axis=0);top=float(np.percentile(P[:,2],95))
        X=P[:,:2]-P[:,:2].mean(0);w,V=np.linalg.eigh(X.T@X/len(X));short=V[:,0]
    else:
        ctr=np.array([0.527,0.099,floorz+0.03]);top=floorz+0.055;short=np.array([-0.64,0.77])
    print('bottle ctr',np.round(ctr,3),'top',round(top,3),'short',np.round(short,2),'floor',round(floorz,3))
    a=np.arctan2(short[1],short[0])-np.pi/2;a=(a+np.pi/2)%np.pi-np.pi/2
    q=quat_mul(np.array([np.cos(a/2),0,0,np.sin(a/2)]),np.array([0.,1.,0.,0.]));q/=np.linalg.norm(q)
    gz=max(top-0.025,floorz+0.012)
    open_gripper()
    goto_pose(np.array([ctr[0],ctr[1],gz]),q,z_approach=0.15)
    close_gripper()
    goto_pose(np.array([ctr[0],ctr[1],gz+0.25]),q)
    g=float(get_observation()['robot_cartesian_pos'][7]);print('gap after lift',g)
    if g>0.03:
        goto_pose(np.array([bc[0],bc[1],rim+0.2]),q)
        goto_pose(np.array([bc[0],bc[1],rim+0.07]),q)
        open_gripper()
        goto_pose(np.array([bc[0],bc[1],rim+0.25]),q)
    else:
        open_gripper();print('empty close')
    goto_home_joint_position()
    obs=get_observation();rgb=np.asarray(obs['agentview']['images']['rgb']).astype(int)
    R,G,B=rgb[...,0],rgb[...,1],rgb[...,2]
    m2=roi&(R>120)&(R-B>60)&(R-G>30);print('orange px left in floor ROI after',int(m2.sum()))
except Exception as e:
    print('manual exc',e)
    try: goto_home_joint_position()
    except Exception as e2: print('home exc',e2)

# Code block 8
import numpy as np
# The bbq bottle (orange body, grey cap) still lies on the floor left of the basket, and the last close was empty.
# New plan: agentview estimate, then hover and refine xy with the wrist camera, then check the hand reach,
# correct it, and grasp a bit higher (z 0.022) to avoid floor contact.
def quat_mul(a,b):
    w1,x1,y1,z1=a;w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])
def bp(cam,mask):
    D=np.asarray(cam['images']['depth'],dtype=np.float64)
    if D.ndim==3: D=D[...,0]
    K=np.asarray(cam['intrinsics']);T=np.asarray(cam['pose_mat'])
    v,u=np.nonzero(mask);z=D[v,u]
    pc=np.stack([(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z],1)
    P=(T[:3,:3]@pc.T).T+T[:3,3]
    return P[np.isfinite(P).all(1)]
def orange(rgb):
    rgb=rgb.astype(int);R,G,B=rgb[...,0],rgb[...,1],rgb[...,2]
    return (R>120)&(R-B>60)&(R-G>30)
BC=np.array([0.673,0.261]);RIM=0.141
try:
    obs=get_observation();a=obs['agentview'];H,W=a['images']['rgb'].shape[:2]
    roi=np.zeros((H,W),bool);roi[225:300,410:500]=True
    P=bp(a,roi&orange(np.asarray(a['images']['rgb'])))
    ctr=np.median(P,axis=0)
    X=P[:,:2]-P[:,:2].mean(0);w,V=np.linalg.eigh(X.T@X/len(X));short=V[:,0]
    print('agent ctr',np.round(ctr,3),'n',len(P))
    ang=np.arctan2(short[1],short[0])-np.pi/2;ang=(ang+np.pi/2)%np.pi-np.pi/2
    q=quat_mul(np.array([np.cos(ang/2),0,0,np.sin(ang/2)]),np.array([0.,1.,0.,0.]));q/=np.linalg.norm(q)
    open_gripper()
    goto_pose(np.array([ctr[0],ctr[1],0.17]),q)
    obs=get_observation();wc=obs['robot0_eye_in_hand']
    Pw=bp(wc,orange(np.asarray(wc['images']['rgb'])))
    if len(Pw): Pw=Pw[(Pw[:,2]<0.08)&(Pw[:,2]>-0.01)&(np.linalg.norm(Pw[:,:2]-ctr[:2],axis=1)<0.1)&(np.linalg.norm(Pw[:,:2]-BC,axis=1)>0.09)]
    print('wrist pts',len(Pw))
    if len(Pw)>=30:
        wctr=np.median(Pw,axis=0);print('wrist ctr',np.round(wctr,3))
        if np.linalg.norm(wctr[:2]-ctr[:2])<0.05: ctr[:2]=wctr[:2]
    tgt=np.array([ctr[0],ctr[1],0.022])
    goto_pose(tgt,q,z_approach=0.12)
    # reach check using hand pose relative to the hover reference
    obs=get_observation();hp=np.asarray(obs['robot_cartesian_pos'][:3])
    print('hand xy',np.round(hp[:2],3),'target',np.round(tgt[:2],3))
    err=tgt[:2]-hp[:2]
    if 0.005<np.linalg.norm(err)<0.04:
        goto_pose(np.array([tgt[0]+err[0],tgt[1]+err[1],tgt[2]]),q)
        hp=np.asarray(get_observation()['robot_cartesian_pos'][:3]);print('corrected hand xy',np.round(hp[:2],3))
    close_gripper()
    goto_pose(np.array([ctr[0],ctr[1],0.27]),q)
    g=float(get_observation()['robot_cartesian_pos'][7]);print('gap after lift',g)
    if g>0.03:
        goto_pose(np.array([BC[0],BC[1],RIM+0.2]),q)
        goto_pose(np.array([BC[0],BC[1],RIM+0.06]),q)
        open_gripper()
        goto_pose(np.array([BC[0],BC[1],RIM+0.25]),q)
    else:
        open_gripper();print('empty close')
    goto_home_joint_position()
    obs=get_observation()
    print('orange px left in floor ROI',int((roi&orange(np.asarray(obs['agentview']['images']['rgb']))).sum()))
except Exception as e:
    print('manual exc',e)
    try: goto_home_joint_position()
    except Exception as e2: print('home exc',e2)