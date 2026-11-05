# Code block 0
import numpy as np
name='small_farm_fresh_butter_box';cont='basket'
r=execute_option('pick',{'object':name})
ctx=r.get('data',{}).get('context',{}) or {}
ev=ctx.get('model_evidence') or {}
st=r.get('status')
print('pick',st,ev.get('mode'),ev.get('opening'),ctx.get('tried'),str(r.get('termination_reason'))[:150])
if st=='command_completed' or (st=='unknown' and ev.get('mode')!='low_elsewhere'):
    ctx['container']=cont
    r2=execute_option('place',ctx)
    c2=r2.get('data',{}).get('context',{}) or {}
    s2=get_abstract_state(c2).get('abstract_state',{}) or {}
    print('place',r2.get('status'),s2.get('placed_in_container'),s2.get('placement_reason'),str(r2.get('termination_reason'))[:120])
elif st=='unknown':
    p=np.array(get_observation()['robot_cartesian_pos'])
    goto_pose(np.array([p[0],p[1],max(p[2]-0.15,0.15)]),p[3:7])
    open_gripper();goto_home_joint_position()
    print('released wrong object; rename and re-pick')
else:
    s=get_abstract_state(ctx).get('abstract_state',{}) or {}
    print('ext',s.get('object_extent_xy'),s.get('object_height'),s.get('object_shift'),s.get('grasp_ladder'))

# Code block 1
import numpy as np
# Both geo grasps closed empty and knocked the box flat (~1.8 cm tall, 3.5x7.2 cm footprint).
# Re-pick from home with a fresh context, a more specific visual name, and the low grasp source.
name='orange_farm_fresh_butter_box_lying_flat';cont='basket'
r=execute_option('pick',{'object':name,'prefer':'geo_low'})
ctx=r.get('data',{}).get('context',{}) or {}
ev=ctx.get('model_evidence') or {}
st=r.get('status')
print('pick',st,ev.get('mode'),ev.get('opening'),ctx.get('tried'),str(r.get('termination_reason'))[:150])
if st=='command_completed' or (st=='unknown' and ev.get('mode')!='low_elsewhere'):
    ctx['container']=cont
    r2=execute_option('place',ctx)
    c2=r2.get('data',{}).get('context',{}) or {}
    s2=get_abstract_state(c2).get('abstract_state',{}) or {}
    print('place',r2.get('status'),s2.get('placed_in_container'),s2.get('placement_reason'),str(r2.get('termination_reason'))[:120])
elif st=='unknown':
    # A different object is probably held: lower it, release, go home.
    p=np.array(get_observation()['robot_cartesian_pos'])
    goto_pose(np.array([p[0],p[1],max(p[2]-0.15,0.15)]),p[3:7])
    open_gripper();goto_home_joint_position()
    print('released wrong object; rename and re-pick')
else:
    s=get_abstract_state(ctx).get('abstract_state',{}) or {}
    print('ext',s.get('object_extent_xy'),s.get('object_height'),s.get('object_shift'),s.get('grasp_ladder'))

# Code block 2
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')

# Code block 3
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')

# Code block 4
import numpy as np
cont='basket'
# Neither pick result printed last time, so measure both small flat boxes first to ground the butter.
cands=['orange_and_white_butter_box','small_orange_box_lying_flat','dark_red_box']
meas={}
for c in cands:
    try:
        d=get_object_3d_points_and_masks_from_language(c)
        pts=d.get('points_3d')
        if pts is not None and len(pts)>20:
            o=get_oriented_bounding_box_from_3d_points(np.asarray(pts))
            meas[c]=(np.round(o['center'],3),np.round(np.sort(o['extent']),3),round(float(d.get('agentview_score') or 0),2))
        else:
            meas[c]=None
    except Exception as e:
        meas[c]='err '+str(e)[:60]
    print('meas',c,meas[c])

name='orange_and_white_butter_box'
if not isinstance(meas.get(name),tuple):
    name='small_orange_box_lying_flat'
r=execute_option('pick',{'object':name,'prefer':'sampled'})
ctx=r.get('data',{}).get('context',{}) or {}
ev=ctx.get('model_evidence') or {}
st=r.get('status')
print('pick',name,st,ev.get('mode'),ev.get('opening'),ctx.get('tried'),str(r.get('termination_reason'))[:150])
if st=='command_completed' or (st=='unknown' and ev.get('mode')!='low_elsewhere'):
    ctx['container']=cont
    r2=execute_option('place',ctx)
    c2=r2.get('data',{}).get('context',{}) or {}
    s2=get_abstract_state(c2).get('abstract_state',{}) or {}
    print('place',r2.get('status'),s2.get('placed_in_container'),s2.get('placement_reason'),str(r2.get('termination_reason'))[:120])
elif st=='unknown':
    p=np.array(get_observation()['robot_cartesian_pos'])
    goto_pose(np.array([p[0],p[1],max(p[2]-0.15,0.15)]),p[3:7])
    open_gripper();goto_home_joint_position()
    print('released wrong object; rename and re-pick')
else:
    s=get_abstract_state(ctx).get('abstract_state',{}) or {}
    print('ext',s.get('object_extent_xy'),s.get('object_height'),s.get('object_shift'),s.get('grasp_ladder'))