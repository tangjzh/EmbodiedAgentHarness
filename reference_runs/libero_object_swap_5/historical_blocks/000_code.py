# Code block 0
c={'object':'tomato_sauce_can','container':'basket','max_dim':0.12,'min_dim':0.05,'alt_name':'red_labeled_tomato_sauce_can_with_silver_top'}
r=execute_option('pick',c);pc=r['data']['context'];s=get_abstract_state(pc)['abstract_state']
print('pick',r['status'],r.get('termination_reason'),s.get('grounding'),s.get('lying'),s.get('toppled'),s.get('tried'),s.get('first_source'),s.get('failure_mode'))
gap=float(get_observation()['robot_cartesian_pos'][7]);print('gap',gap)
if r['status']!='command_completed' and not (r['status']=='unknown' and gap>0.04):
    g=s.get('grounding')
    nc={'object':'red_tomato_sauce_can','container':'basket','max_dim':0.12,'min_dim':0.05,'alt_name':'cylindrical_red_can_behind_orange_juice_carton'}
    for k in ('container_points','container_pose'):
        if pc.get(k) is not None: nc[k]=pc[k]
    if g in ('missing','fragment','off_support','size_mismatch'):
        pass
    elif s.get('lying') is True or s.get('toppled') is True:
        nc['prefer']='geo_axis'
    else:
        nc['tried_in']=[t for t in (list(s.get('tried') or [])+[s.get('first_source')]) if t]
    r=execute_option('pick',nc);pc=r['data']['context'];s=get_abstract_state(pc)['abstract_state']
    print('repick',r['status'],r.get('termination_reason'),s.get('grounding'),s.get('tried'),s.get('failure_mode'))
    gap=float(get_observation()['robot_cartesian_pos'][7]);print('gap',gap)
if r['status']=='command_completed' or (r['status']=='unknown' and gap>0.04):
    r2=execute_option('place',pc);p2=r2['data']['context']
    s2=get_abstract_state(p2)['abstract_state']
    print('place',r2['status'],r2.get('termination_reason'),s2.get('placement_reason'),s2.get('placed_in_container'))
    if r2['status']=='unknown':
        v=execute_option('verify',p2);vs=get_abstract_state(v['data']['context'])['abstract_state']
        print('verify',v['status'],v.get('termination_reason'),vs.get('placed_in_container'),vs.get('placement_reason'))

# Code block 1
import numpy as np
# Place reported outside_xy, but the can is not visible on the floor in either view and may be hidden by the basket walls.
# Do a motion-free verify, print compact evidence, and decide on a re-pick next turn.
p2c = r2['data']['context']
keys = list(p2c.keys())
print('ctx keys', keys)
for k in ('container_pose', 'final_object', 'final_object_pose', 'placement_reason', 'placed_in_container'):
    v = p2c.get(k)
    if v is None:
        continue
    try:
        if isinstance(v, (list, tuple)):
            print(k, [np.round(np.asarray(x, dtype=float), 3).tolist() for x in v])
        else:
            print(k, np.round(np.asarray(v, dtype=float), 3).tolist())
    except Exception:
        print(k, str(v)[:200])
cp = p2c.get('container_points')
if cp is not None:
    cp = np.asarray(cp)
    print('container pts min', np.round(cp.min(0), 3).tolist(), 'max', np.round(cp.max(0), 3).tolist())
obs = get_observation()
print('ee', np.round(obs['robot_cartesian_pos'], 3).tolist())
v = execute_option('verify', p2c)
vc = v['data']['context']
vs = get_abstract_state(vc)['abstract_state']
print('verify', v['status'], v.get('termination_reason'), vs.get('placed_in_container'), vs.get('placement_reason'), vs.get('object_center'))
for k in ('final_object', 'final_object_pose', 'object_pose'):
    val = vc.get(k)
    if val is not None:
        print('v', k, str(val)[:200])

# Code block 2
import numpy as np
# The image shows the tomato sauce is a dark red bottle with an orange cap near the robot base.
# It is still on the floor and the basket is empty. Earlier picks used "can" names, which may have been the wrong grounding.
# Do one bounded re-pick with a bottle description, reusing the container geometry.
try:
    prev = r2['data']['context']
except Exception:
    prev = {}
try:
    ps = get_abstract_state(prev)['abstract_state']
except Exception:
    ps = {}
nc = {'object': 'tomato_sauce_bottle', 'container': 'basket', 'max_dim': 0.15, 'min_dim': 0.04,
      'alt_name': 'dark_red_tomato_sauce_bottle_with_orange_cap'}
for k in ('container_points', 'container_pose'):
    if prev.get(k) is not None:
        nc[k] = prev[k]
if ps.get('lying') is True or ps.get('toppled') is True:
    nc['prefer'] = 'geo_axis'
print('prev state', ps.get('tried'), ps.get('first_source'), ps.get('lying'), ps.get('toppled'))
try:
    r = execute_option('pick', nc)
    pc = r['data']['context']
    s = get_abstract_state(pc)['abstract_state']
    print('pick', r['status'], str(r.get('termination_reason'))[:150], s.get('grounding'), s.get('lying'),
          s.get('toppled'), s.get('tried'), s.get('first_source'), s.get('failure_mode'))
    gap = float(get_observation()['robot_cartesian_pos'][7])
    print('gap', gap)
    if r['status'] == 'command_completed' or (r['status'] == 'unknown' and gap > 0.04):
        r2 = execute_option('place', pc)
        p2 = r2['data']['context']
        s2 = get_abstract_state(p2)['abstract_state']
        print('place', r2['status'], str(r2.get('termination_reason'))[:150], s2.get('placement_reason'),
              s2.get('placed_in_container'))
        if r2['status'] == 'unknown':
            v = execute_option('verify', p2)
            vs = get_abstract_state(v['data']['context'])['abstract_state']
            print('verify', v['status'], vs.get('placed_in_container'), vs.get('placement_reason'))
    else:
        print('pick did not attach; inspect images next turn')
except Exception as e:
    print('error', str(e)[:200])
print('ee', np.round(get_observation()['robot_cartesian_pos'], 3).tolist())