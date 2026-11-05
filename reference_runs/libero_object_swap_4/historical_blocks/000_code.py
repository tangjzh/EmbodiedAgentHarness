# Code block 0
ctx={'object':'large_tomato_ketchup_bottle','container':'basket'}
r=execute_option('pick',ctx)
pc=r['data']['context']
s=get_abstract_state(pc)['abstract_state']
print('pick1',r['status'],r.get('termination_reason'),s.get('object_posture'),s.get('partial_cloud'),s.get('toppled'),s.get('failure_mode'),s.get('retry_source'))

def carry(prev,extra):
    c={'object':'large_tomato_ketchup_bottle','container':'basket'}
    c.update(extra)
    for k in ('container_points','container_pose'):
        if prev.get(k) is not None:
            c[k]=prev[k]
    return c

if r['status']=='error' and 'rasp' in str(r.get('termination_reason')):
    r=execute_option('pick',carry(pc,{'sampler_failed':True}))
    pc=r['data']['context']
    s=get_abstract_state(pc)['abstract_state']
    print('pick2',r['status'],r.get('termination_reason'),s.get('object_posture'),s.get('toppled'),s.get('failure_mode'))
elif r['status']=='handoff':
    r=execute_option('pick',carry(pc,{'prefer':'geo'}))
    pc=r['data']['context']
    s=get_abstract_state(pc)['abstract_state']
    print('pick2',r['status'],r.get('termination_reason'),s.get('object_posture'),s.get('toppled'),s.get('failure_mode'))

if r['status'] in ('command_completed','unknown'):
    r2=execute_option('place',pc)
    c2=r2['data']['context']
    s2=get_abstract_state(c2)['abstract_state']
    print('place',r2['status'],r2.get('termination_reason'),s2.get('placed_in_container'),s2.get('placement_reason'))
    if r2['status']=='unknown':
        v=execute_option('verify',c2)
        sv=get_abstract_state(v['data']['context'])['abstract_state']
        print('verify',v['status'],sv.get('placed_in_container'),sv.get('placement_reason'))
else:
    print('pick not successful; will re-plan after observing')

# Code block 1
# Pick 1 grounded the wrong object: a small dark sauce bottle was lifted and dropped on the basket rim.
# The orange "Tomato Ketchup" bottle with the white cap is still upright on the floor, in front of and left of the basket.
# Changes now: a more specific object description, carried-over container geometry, and a pose check before picking.
import numpy as np
prev = pc
obj = 'orange_tomato_ketchup_bottle_with_white_cap'

def carry(prev, extra):
    c = {'object': obj, 'container': 'basket'}
    c.update(extra)
    for k in ('container_points', 'container_pose'):
        if prev.get(k) is not None:
            c[k] = prev[k]
    return c

p, _ = get_object_pose(obj, use_multiview=True)
print('target pose check', None if p is None else np.round(p, 3))
cp = prev.get('container_pose')
try:
    print('container pose', None if cp is None else np.round(np.asarray(cp[0] if isinstance(cp, (tuple, list)) else cp), 3))
except Exception as e:
    print('container pose print err', e)

r = execute_option('pick', carry(prev, {}))
pc = r['data']['context']
s = get_abstract_state(pc)['abstract_state']
print('pickA', r['status'], str(r.get('termination_reason'))[:150], s.get('object_posture'), s.get('toppled'), s.get('failure_mode'), np.round(np.asarray(s.get('object_center') or [0, 0, 0]), 3))

if r['status'] == 'error' and 'rasp' in str(r.get('termination_reason')):
    r = execute_option('pick', carry(pc, {'sampler_failed': True}))
    pc = r['data']['context']
    s = get_abstract_state(pc)['abstract_state']
    print('pickB', r['status'], str(r.get('termination_reason'))[:150], s.get('failure_mode'))
elif r['status'] == 'handoff':
    r = execute_option('pick', carry(pc, {'prefer': 'geo'}))
    pc = r['data']['context']
    s = get_abstract_state(pc)['abstract_state']
    print('pickB', r['status'], str(r.get('termination_reason'))[:150], s.get('failure_mode'))

if r['status'] == 'command_completed':
    r2 = execute_option('place', pc)
    c2 = r2['data']['context']
    s2 = get_abstract_state(c2)['abstract_state']
    print('place', r2['status'], str(r2.get('termination_reason'))[:150], s2.get('placed_in_container'), s2.get('placement_reason'))
    if r2['status'] == 'unknown':
        v = execute_option('verify', c2)
        sv = get_abstract_state(v['data']['context'])['abstract_state']
        print('verify', v['status'], sv.get('placed_in_container'), sv.get('placement_reason'))
elif r['status'] == 'unknown':
    # Fingers are open but attachment is ambiguous: inspect the images before deciding.
    obs = get_observation()
    print('unknown pick; hand pose', np.round(obs['robot_cartesian_pos'], 3), '- inspect image next step before place')
else:
    print('pick not successful; re-plan after observing')

# Code block 2
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')

# Code block 3
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')

# Code block 4
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')

# Code block 5
import numpy as np
obj = 'orange_tomato_ketchup_bottle_with_white_cap'

# Hand state now: the image shows the orange ketchup bottle held near the basket.
obs = get_observation()
hp = obs['robot_cartesian_pos']
print('hand xyz', np.round(hp[:3], 3), 'quat', np.round(hp[3:7], 3), 'gripper', round(float(hp[7]), 3))

# Rebuild a context and carry over container geometry if a previous one exists.
try:
    prev = pc
except NameError:
    prev = {}
ctx = {'object': obj, 'container': 'basket'}
for k in ('container_points', 'container_pose', 'grasp', 'object_points', 'object_pose'):
    try:
        if prev.get(k) is not None and (k.startswith('container') or prev.get('object') == obj):
            ctx[k] = prev[k]
    except Exception:
        pass
print('ctx keys', sorted(ctx.keys()))

def try_pick(extra):
    c = {'object': obj, 'container': 'basket'}
    c.update(extra)
    for k in ('container_points', 'container_pose'):
        if ctx.get(k) is not None:
            c[k] = ctx[k]
    r = execute_option('pick', c)
    s = get_abstract_state(r['data']['context'])['abstract_state']
    print('pick', r['status'], str(r.get('termination_reason'))[:150], s.get('object_posture'), s.get('failure_mode'))
    return r

def do_place(c):
    r2 = execute_option('place', c)
    c2 = r2['data']['context']
    s2 = get_abstract_state(c2)['abstract_state']
    print('place', r2['status'], str(r2.get('termination_reason'))[:150], s2.get('placed_in_container'), s2.get('placement_reason'))
    if r2['status'] in ('unknown', 'command_completed'):
        v = execute_option('verify', c2)
        sv = get_abstract_state(v['data']['context'])['abstract_state']
        print('verify', v['status'], sv.get('placed_in_container'), sv.get('placement_reason'))
    return r2

holding = float(hp[7]) > 0.05  # The fingers are not fully closed, so the gripper is likely holding the bottle.
if holding:
    r2 = do_place(ctx)
    if r2['status'] == 'option_terminated':
        r = try_pick({'prefer': 'geo'})
        if r['status'] in ('command_completed', 'unknown'):
            do_place(r['data']['context'])
else:
    print('gripper closed/empty; re-picking')
    open_gripper(); goto_home_joint_position()
    r = try_pick({'prefer': 'geo'})
    if r['status'] == 'error' and 'rasp' in str(r.get('termination_reason')):
        r = try_pick({'sampler_failed': True})
    if r['status'] in ('command_completed', 'unknown'):
        do_place(r['data']['context'])
print('final gripper', round(float(get_observation()['robot_cartesian_pos'][7]), 3))

# Code block 6
import numpy as np
# The basket was knocked onto its side, so the old container geometry is stale. Re-measure it fresh.
# The orange ketchup bottle now lies on the floor at the basket opening.
# Plan: measure both objects compactly, then make one pick attempt with a new name, prefer='geo' and fresh container geometry.
obj = 'orange_ketchup_bottle_lying_on_floor'

def compact(name):
    try:
        d = get_object_3d_points_and_masks_from_language(name, use_multiview=False)
        pts = d.get('points_3d')
        if pts is None or len(pts) < 10:
            print(name, 'few points'); return None
        ob = get_oriented_bounding_box_from_3d_points(np.asarray(pts, dtype=np.float64))
        print(name, 'n', len(pts), 'center', np.round(ob['center'], 3), 'extent', np.round(ob['extent'], 3),
              'zmin/zmax', round(float(np.min(pts[:, 2])), 3), round(float(np.max(pts[:, 2])), 3),
              'score', round(float(d.get('agentview_score') or 0), 2))
        return ob
    except Exception as e:
        print(name, 'err', str(e)[:100]); return None

bb = compact('basket')
kb = compact(obj)

# Fresh context, so the pick re-measures the container (no stale carry-over).
c = {'object': obj, 'container': 'basket', 'prefer': 'geo'}
r = execute_option('pick', c)
pc = r['data']['context']
s = get_abstract_state(pc)['abstract_state']
print('pick', r['status'], str(r.get('termination_reason'))[:120], s.get('object_posture'), s.get('toppled'), s.get('failure_mode'), r['data'].get('model_evidence') if isinstance(r.get('data'), dict) else None)
if r['status'] == 'error' and 'rasp' in str(r.get('termination_reason')):
    r = execute_option('pick', {'object': obj, 'container': 'basket', 'sampler_failed': True})
    pc = r['data']['context']
    print('pick2', r['status'], str(r.get('termination_reason'))[:120])

if r['status'] in ('command_completed', 'unknown'):
    hp = get_observation()['robot_cartesian_pos']
    print('hand', np.round(hp[:3], 3), 'grip', round(float(hp[7]), 3))
    if float(hp[7]) > 0.05:
        r2 = execute_option('place', pc)
        c2 = r2['data']['context']
        s2 = get_abstract_state(c2)['abstract_state']
        print('place', r2['status'], str(r2.get('termination_reason'))[:120], s2.get('placed_in_container'), s2.get('placement_reason'))
        if r2['status'] == 'unknown':
            v = execute_option('verify', c2)
            sv = get_abstract_state(v['data']['context'])['abstract_state']
            print('verify', v['status'], sv.get('placed_in_container'), sv.get('placement_reason'))
    else:
        print('gripper closed; not holding')
        open_gripper(); goto_home_joint_position()
else:
    print('pick not successful; inspect images next step')
print('final grip', round(float(get_observation()['robot_cartesian_pos'][7]), 3))

# Code block 7
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')

# Code block 8
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')