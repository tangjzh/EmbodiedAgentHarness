# Code block 0
ctx={'object':'cream_cheese_box','container':'basket'}
r=execute_option('pick',ctx);pc=r['data']['context']
s=get_abstract_state(pc)['abstract_state']
print('pick',r['status'],s.get('failure_mode'),s.get('close_gap'),s.get('grasp_primary_source'))
if r['status']=='handoff':
    c={'object':'cream_cheese_box','container':'basket','prefer':'geo'}
    for k in ('container_points','container_pose'):
        if pc.get(k) is not None:
            c[k]=pc[k]
    r=execute_option('pick',c);pc=r['data']['context']
    s=get_abstract_state(pc)['abstract_state']
    print('repick',r['status'],s.get('failure_mode'),s.get('close_gap'),s.get('grasp_primary_source'))
if r['status']=='command_completed':
    r2=execute_option('place',pc);c2=r2['data']['context']
    s2=get_abstract_state(c2)['abstract_state']
    print('place',r2['status'],s2.get('hang'),s2.get('hang_source'),s2.get('release_mode'),s2.get('placed_in_container'),s2.get('placement_reason'),s2.get('final_top'),s2.get('final_bottom'))
    if r2['status']=='unknown':
        v=execute_option('verify',c2)
        sv=get_abstract_state(v['data']['context'])['abstract_state']
        print('verify',v['status'],sv.get('placed_in_container'),sv.get('placement_reason'))
elif r['status']=='unknown':
    print('pick unknown; inspect images before place')

# Code block 1
# The GraspNet sampler errored, so retry the pick with sampler_failed=True and geometric grasps.
# Reuse the container geometry that was already measured.
c={'object':'cream_cheese_box','container':'basket','sampler_failed':True,'prefer':'geo'}
for k in ('container_points','container_pose'):
    if pc.get(k) is not None:
        c[k]=pc[k]
r=execute_option('pick',c);pc=r['data']['context']
s=get_abstract_state(pc)['abstract_state']
print('pick',r['status'],r.get('termination_reason'),s.get('failure_mode'),s.get('close_gap'),s.get('grasp_primary_source'))

# Bounded recovery: on handoff, re-pick once using the alternate geometric grasp (different yaw)
if r['status']=='handoff':
    c={'object':'cream_cheese_box','container':'basket','sampler_failed':True,'prefer':'geo_alt'}
    for k in ('container_points','container_pose'):
        if pc.get(k) is not None:
            c[k]=pc[k]
    r=execute_option('pick',c);pc=r['data']['context']
    s=get_abstract_state(pc)['abstract_state']
    print('repick',r['status'],r.get('termination_reason'),s.get('failure_mode'),s.get('close_gap'),s.get('grasp_primary_source'))

# Place only after a confirmed pick; if place is unknown, run verify
if r['status']=='command_completed':
    r2=execute_option('place',pc);c2=r2['data']['context']
    s2=get_abstract_state(c2)['abstract_state']
    print('place',r2['status'],r2.get('termination_reason'),s2.get('hang'),s2.get('hang_source'),s2.get('release_mode'),s2.get('placed_in_container'),s2.get('placement_reason'),s2.get('final_top'),s2.get('final_bottom'))
    if r2['status']=='unknown':
        v=execute_option('verify',c2)
        sv=get_abstract_state(v['data']['context'])['abstract_state']
        print('verify',v['status'],sv.get('placed_in_container'),sv.get('placement_reason'))
elif r['status']=='unknown':
    print('pick unknown; inspect images before place')

# Code block 2
# The images show the butter box in the hand, not the cream cheese. Release it and go home.
r=execute_option('release_home',{})
print('release_home',r['status'])

def mk(obj,prefer):
    c={'object':obj,'container':'basket','sampler_failed':True,'prefer':prefer}
    for k in ('container_points','container_pose'):
        if pc.get(k) is not None:
            c[k]=pc[k]
    return c

# Re-pick with a more specific description so segmentation finds the small blue cream cheese box.
obj='small_blue_cream_cheese_box'
r=execute_option('pick',mk(obj,'geo'));pc2=r['data']['context']
s=get_abstract_state(pc2)['abstract_state']
print('pick',r['status'],s.get('failure_mode'),s.get('close_gap'),s.get('object_center'),s.get('object_height'))
if r['status']=='handoff':
    r=execute_option('pick',mk(obj,'geo_alt'));pc2=r['data']['context']
    s=get_abstract_state(pc2)['abstract_state']
    print('repick',r['status'],s.get('failure_mode'),s.get('close_gap'),s.get('object_center'))

if r['status']=='command_completed':
    r2=execute_option('place',pc2);c2=r2['data']['context']
    s2=get_abstract_state(c2)['abstract_state']
    print('place',r2['status'],r2.get('termination_reason'),s2.get('hang'),s2.get('hang_source'),s2.get('release_mode'),s2.get('placed_in_container'),s2.get('placement_reason'),s2.get('final_top'),s2.get('final_bottom'))
    if r2['status']=='unknown':
        v=execute_option('verify',c2)
        sv=get_abstract_state(v['data']['context'])['abstract_state']
        print('verify',v['status'],sv.get('placed_in_container'),sv.get('placement_reason'))
elif r['status']=='unknown':
    print('pick unknown; inspect images before place')

# Code block 3
import numpy as np
# Diagnosis: the first cloud for 'small_blue_cream_cheese_box' was at (0.48,-0.24). That is image-left and far, which matches the
# blue box beside the tomato can. pose0 instead locked onto the butter box near the camera at (0.75,0.03), so the grasp closed empty there.
# Recovery: re-measure the blue box with the agentview camera only (the wrist SAM3 failed). Then grasp it directly and use the place option.
ctx_base={'object':'cream_cheese_box','container':'basket'}
for k in ('container_points','container_pose'):
    if pc.get(k) is not None:
        ctx_base[k]=pc[k]

target=np.array([0.4806,-0.2387]); gz=0.0115; quat=np.array([0.0,0.0173,0.99985,0.0])
try:
    d=get_object_3d_points_and_masks_from_language('small_blue_cream_cheese_box',use_multiview=False)
    P=np.asarray(d['points_3d'])
    print('seg n',len(P),'score',d.get('agentview_score'))
    if len(P)>20:
        c=np.median(P,axis=0)
        print('center',c,'zmin',P[:,2].min(),'zmax',P[:,2].max())
        if np.linalg.norm(c[:2]-target)<0.1:
            target=c[:2]
            obb=get_oriented_bounding_box_from_3d_points(P)
            R=obb['R'];e=obb['extent']
            # Close the fingers across the shorter horizontal axis.
            ax=[i for i in range(3) if abs(R[2,i])<0.7]
            if len(ax)>=2:
                short=min(ax,key=lambda i:e[i])
                v=R[:,short];yaw=np.arctan2(v[1],v[0])
                # Gripper pointing down: q=(0,cos(yaw/2),sin(yaw/2),0) puts the finger axis along yaw.
                quat=np.array([0.0,np.cos(yaw/2),np.sin(yaw/2),0.0])
            gz=max(P[:,2].min()+0.01,0.008)
except Exception as ex:
    print('seg err',ex)
print('grasp at',target,gz,quat)

open_gripper()
goto_pose(np.array([target[0],target[1],gz]),quat,z_approach=0.12)
close_gripper()
g=get_observation()['robot_cartesian_pos'][-1]
print('gap after close',g)
goto_pose(np.array([target[0],target[1],0.22]),quat)
g2=get_observation()['robot_cartesian_pos'][-1]
print('gap after lift',g2)
if g2>0.08:
    r2=execute_option('place',dict(ctx_base));c2=r2['data']['context']
    s2=get_abstract_state(c2)['abstract_state']
    print('place',r2['status'],r2.get('termination_reason'),s2.get('hang'),s2.get('release_mode'),s2.get('placed_in_container'),s2.get('placement_reason'),s2.get('final_top'),s2.get('final_bottom'))
    if r2['status']=='unknown':
        v=execute_option('verify',c2)
        sv=get_abstract_state(v['data']['context'])['abstract_state']
        print('verify',v['status'],sv.get('placed_in_container'),sv.get('placement_reason'))
else:
    print('empty grasp; releasing and going home')
    execute_option('release_home',{})

# Code block 4
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')

# Code block 5
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')

# Code block 6
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')

# Code block 7
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')

# Code block 8
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')