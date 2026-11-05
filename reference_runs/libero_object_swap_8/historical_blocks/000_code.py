# Code block 0
ctx={'object':'chocolate_pudding_box','container':'basket'}
single=False
try:
    r=execute_option('pick',ctx)
except Exception as e:
    r={'status':'error','termination_reason':str(e),'data':{'context':ctx}}
t=str(r.get('termination_reason'))
print('pick1',r['status'],t[:150])
if r['status']=='error' and ('SAM' in t or 'eye_in_hand' in t or 'wrist' in t):
    single=True
    ctx={'object':'chocolate_pudding_box','container':'basket','single_view':True}
    try:
        r=execute_option('pick',ctx)
    except Exception as e:
        r={'status':'error','termination_reason':str(e),'data':{'context':ctx}}
    t=str(r.get('termination_reason'))
    print('pick1sv',r['status'],t[:150])
pc=r['data']['context']
try:
    s=get_abstract_state(pc)['abstract_state']
except Exception as e:
    s={}
    print('abs err',str(e)[:100])
print('state',s.get('failure_mode'),s.get('wrong_object_suspected'),s.get('reject_xy'),s.get('reject_hit'))
rej=list(s.get('reject_xy') or [])

if r['status']=='option_terminated' and s.get('wrong_object_suspected'):
    sd=execute_option('set_down',pc)
    rej=get_abstract_state(sd['data']['context'])['abstract_state'].get('reject_xy') or rej
    print('set_down',sd['status'],rej)
    c={'object':'small_brown_chocolate_pudding_box_lying_flat','container':'basket','reject_xy':rej}
    if single: c['single_view']=True
    r=execute_option('pick',c)
    pc=r['data']['context']
    s=get_abstract_state(pc)['abstract_state']
    print('repick',r['status'],str(r.get('termination_reason'))[:120],s.get('failure_mode'),s.get('reject_hit'))
elif r['status']=='handoff':
    c={'object':'chocolate_pudding_box','container':'basket','prefer':'geo','reject_xy':rej}
    if single: c['single_view']=True
    r=execute_option('pick',c)
    pc=r['data']['context']
    s=get_abstract_state(pc)['abstract_state']
    print('repick_geo',r['status'],str(r.get('termination_reason'))[:120],s.get('failure_mode'),s.get('reject_hit'))

if r['status']=='command_completed':
    r2=execute_option('place',pc)
    c2=r2['data']['context']
    s2=get_abstract_state(c2)['abstract_state']
    print('place',r2['status'],str(r2.get('termination_reason'))[:120],s2.get('placed_in_container'),s2.get('placement_reason'))
    if r2['status']=='unknown':
        v=execute_option('verify',c2)
        sv=get_abstract_state(v['data']['context'])['abstract_state']
        print('verify',v['status'],sv.get('placed_in_container'),sv.get('placement_reason'))
else:
    print('final pick status',r['status'],'- inspect images before next step')

obs=get_observation()
print('gripper',np.round(obs['robot_cartesian_pos'],3) if 'np' in dir() else obs['robot_cartesian_pos'])

# Code block 1
import numpy as np
# GraspNet service returned a 500 error. Grounding looked correct: Molmo pointed at about (277,334) px, which is the pudding box.
# Retry with a geometric grasp, which does not call GraspNet.
def run_pick(c):
    try:
        r=execute_option('pick',c)
    except Exception as e:
        r={'status':'error','termination_reason':str(e),'data':{'context':c}}
    pc=r['data']['context']
    try:
        s=get_abstract_state(pc)['abstract_state']
    except Exception as e:
        s={}
    print('pick',c.get('prefer'),c.get('single_view'),r['status'],str(r.get('termination_reason'))[:120],
          s.get('failure_mode'),s.get('wrong_object_suspected'),s.get('reject_xy'),s.get('reject_hit'))
    return r,pc,s

single=False
rej=[]
c={'object':'chocolate_pudding_box','container':'basket','prefer':'geo','sampler_failed':True}
r,pc,s=run_pick(c)
t=str(r.get('termination_reason'))
if r['status']=='error' and ('SAM' in t or 'eye_in_hand' in t or 'wrist' in t):
    single=True
    c={'object':'chocolate_pudding_box','container':'basket','prefer':'geo','sampler_failed':True,'single_view':True}
    r,pc,s=run_pick(c)
rej=list(s.get('reject_xy') or [])

if r['status']=='option_terminated' and s.get('wrong_object_suspected'):
    sd=execute_option('set_down',pc)
    rej=get_abstract_state(sd['data']['context'])['abstract_state'].get('reject_xy') or rej
    print('set_down',sd['status'],rej)
    c={'object':'small_brown_chocolate_pudding_box','container':'basket','prefer':'geo','sampler_failed':True,'reject_xy':rej}
    if single: c['single_view']=True
    r,pc,s=run_pick(c)
elif r['status'] in ('handoff','error'):
    c={'object':'chocolate_pudding_box','container':'basket','prefer':'geo_alt','sampler_failed':True,'reject_xy':rej}
    if single: c['single_view']=True
    r,pc,s=run_pick(c)

if r['status']=='command_completed':
    r2=execute_option('place',pc)
    c2=r2['data']['context']
    s2=get_abstract_state(c2)['abstract_state']
    print('place',r2['status'],str(r2.get('termination_reason'))[:120],s2.get('placed_in_container'),s2.get('placement_reason'))
    if r2['status']=='unknown':
        v=execute_option('verify',c2)
        sv=get_abstract_state(v['data']['context'])['abstract_state']
        print('verify',v['status'],sv.get('placed_in_container'),sv.get('placement_reason'))
else:
    print('final pick status',r['status'],'- inspect images')

obs=get_observation()
print('ee',np.round(obs['robot_cartesian_pos'],3))

# Code block 2
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')

# Code block 3
import numpy as np

# The previous run printed nothing useful and the scene did not change.
# The pudding box (front-left brown box) is still on the floor, and the gripper is home and empty.
# Plan: run the option pipeline with a geometric grasp, with every call guarded.
# If that does not reach command_completed, fall back to a manual top-down grasp built from the segmented cloud.

def S(ctx):
    try:
        return get_abstract_state(ctx)['abstract_state']
    except Exception as e:
        print('abs err', str(e)[:80])
        return {}

def run(name, c):
    try:
        r = execute_option(name, c)
    except Exception as e:
        r = {'status': 'error', 'termination_reason': str(e), 'data': {'context': c}}
    cc = (r.get('data') or {}).get('context') or c
    s = S(cc)
    print(name, c.get('prefer'), c.get('single_view'), r.get('status'),
          str(r.get('termination_reason'))[:120],
          s.get('failure_mode'), s.get('wrong_object_suspected'),
          s.get('reject_xy'), s.get('reject_hit'),
          s.get('placed_in_container'), s.get('placement_reason'))
    return r, cc, s

def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])

def gap():
    return float(get_observation()['robot_cartesian_pos'][7])

name = 'chocolate_pudding_box'
single = False
rej = []
placed = False

# 1) Option pipeline with a geometric grasp
c = {'object': name, 'container': 'basket', 'prefer': 'geo', 'sampler_failed': True}
r, pc, s = run('pick', c)
t = str(r.get('termination_reason'))
if r['status'] == 'error' and ('SAM' in t or 'eye_in_hand' in t or 'wrist' in t):
    single = True
    c = {'object': name, 'container': 'basket', 'prefer': 'geo', 'sampler_failed': True, 'single_view': True}
    r, pc, s = run('pick', c)
rej = list(s.get('reject_xy') or [])

# Wrong item held: put it back and re-pick, excluding that spot
if r['status'] == 'option_terminated' and s.get('wrong_object_suspected'):
    sd, sdc, sds = run('set_down', pc)
    rej = sds.get('reject_xy') or rej
    c = {'object': 'small_brown_chocolate_pudding_box', 'container': 'basket',
         'prefer': 'geo_alt', 'sampler_failed': True, 'reject_xy': rej}
    if single:
        c['single_view'] = True
    r, pc, s = run('pick', c)
# Handoff or error: retry once with the alternate geometric grasp
elif r['status'] in ('handoff', 'error'):
    c = {'object': name, 'container': 'basket', 'prefer': 'geo_alt', 'sampler_failed': True, 'reject_xy': rej}
    if single:
        c['single_view'] = True
    r, pc, s = run('pick', c)

if r['status'] == 'command_completed':
    r2, c2, s2 = run('place', pc)
    if r2['status'] == 'command_completed':
        placed = bool(s2.get('placed_in_container'))
    elif r2['status'] == 'unknown':
        v, vc, sv = run('verify', c2)
        placed = bool(sv.get('placed_in_container'))
elif r['status'] == 'unknown':
    print('held_unknown, gap', round(gap(), 3))

# 2) Manual fallback if the option pipeline did not place the box
if not placed and r['status'] not in ('command_completed', 'unknown'):
    try:
        goto_home_joint_position()
        open_gripper()

        # Segment the box and fit an oriented bounding box
        seg = get_object_3d_points_and_masks_from_language('small brown chocolate pudding box', use_multiview=not single)
        P = np.asarray(seg['points_3d'])
        P = P[np.isfinite(P).all(1)]
        print('npts', len(P), 'score', seg.get('agentview_score'))
        obb = get_oriented_bounding_box_from_3d_points(P)
        ctr = np.asarray(obb['center'])
        ext = np.asarray(obb['extent'])
        R = np.asarray(obb['R'])

        # Close the fingers across the shorter horizontal axis of the box
        horiz = [i for i in range(3) if abs(R[2, i]) < 0.7]
        if len(horiz) >= 2:
            i_short = min(horiz, key=lambda i: ext[i])
            ax = R[:, i_short]
            yaw = np.arctan2(ax[1], ax[0])
        else:
            yaw = 0.0
        top = float(np.percentile(P[:, 2], 95))
        print('ctr', np.round(ctr, 3), 'ext', np.round(ext, 3), 'top', round(top, 3), 'yaw', round(yaw, 2))

        # Top-down gripper orientation rotated about world z by yaw
        qd = np.array([0., 0., 1., 0.])
        qz = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
        q = quat_mul(qz, qd)
        q = q / np.linalg.norm(q)
        # Gripper y-axis is the finger closing direction; if the result fails, try yaw + pi/2 next time.
        gpos = np.array([ctr[0], ctr[1], max(top - 0.02, float(P[:, 2].min()) + 0.01)])

        goto_pose(gpos, q, z_approach=0.12)
        close_gripper()
        g = gap()
        print('closed gap', round(g, 3))
        goto_pose(gpos + np.array([0, 0, 0.2]), q)
        g2 = gap()
        print('lifted gap', round(g2, 3))

        if g2 > 0.03:
            # Something is held: carry it to the basket, keeping the grasp orientation
            bp, _ = get_object_pose('basket', use_multiview=not single)
            print('basket', None if bp is None else np.round(bp, 3))
            if bp is not None:
                goto_pose(np.array([bp[0], bp[1], gpos[2] + 0.25]), q)
                goto_pose(np.array([bp[0], bp[1], bp[2] + 0.15]), q)
                open_gripper()
                goto_pose(np.array([bp[0], bp[1], bp[2] + 0.3]), q)
        else:
            print('manual grasp empty')
            open_gripper()
        goto_home_joint_position()

        op, _ = get_object_pose('small brown chocolate pudding box', use_multiview=False)
        bp2, _ = get_object_pose('basket', use_multiview=False)
        print('obj after', None if op is None else np.round(op, 3),
              'basket', None if bp2 is None else np.round(bp2, 3))
    except Exception as e:
        print('manual err', str(e)[:150])

obs = get_observation()
print('ee', np.round(obs['robot_cartesian_pos'], 3))

# Code block 4
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')

# Code block 5
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')

# Code block 6
import numpy as np

# The scene is unchanged and the earlier output was lost, so run again with explicit prints at every step.
print('start', flush=True)

def S(ctx):
    try:
        return get_abstract_state(ctx)['abstract_state']
    except Exception as e:
        print('abs err', str(e)[:80], flush=True)
        return {}

def run(name, c):
    try:
        r = execute_option(name, c)
    except Exception as e:
        r = {'status': 'error', 'termination_reason': str(e), 'data': {'context': c}}
    cc = (r.get('data') or {}).get('context') or c
    s = S(cc)
    print(name, c.get('prefer'), c.get('single_view'), r.get('status'),
          str(r.get('termination_reason'))[:150], s.get('failure_mode'),
          s.get('wrong_object_suspected'), s.get('reject_xy'), s.get('reject_hit'),
          s.get('placed_in_container'), s.get('placement_reason'), flush=True)
    return r, cc, s

name = 'chocolate_pudding_box'
single = False
rej = []
placed = False

c = {'object': name, 'container': 'basket', 'prefer': 'geo', 'sampler_failed': True}
r, pc, s = run('pick', c)
t = str(r.get('termination_reason'))

# A wrist/SAM failure means we switch to agentview-only for the rest of the episode.
if r['status'] == 'error' and ('SAM' in t or 'eye_in_hand' in t or 'wrist' in t):
    single = True
    c = dict(c, single_view=True)
    r, pc, s = run('pick', c)
rej = list(s.get('reject_xy') or [])

if r['status'] == 'option_terminated' and s.get('wrong_object_suspected'):
    # A different item is held: put it back and exclude that spot.
    sd, sdc, sds = run('set_down', pc)
    rej = list(sds.get('reject_xy') or rej)
    c = {'object': 'small_brown_chocolate_pudding_box', 'container': 'basket',
         'prefer': 'geo_alt', 'sampler_failed': True, 'reject_xy': rej}
    if single:
        c['single_view'] = True
    r, pc, s = run('pick', c)
elif r['status'] in ('handoff', 'error'):
    # Hand is home and empty: try the structurally different geometric grasp.
    c = {'object': name, 'container': 'basket', 'prefer': 'geo_alt', 'sampler_failed': True, 'reject_xy': rej}
    if single:
        c['single_view'] = True
    r, pc, s = run('pick', c)

if r['status'] == 'unknown':
    g = float(get_observation()['robot_cartesian_pos'][7])
    print('held_unknown gap', round(g, 3), flush=True)
    # Try placing; place terminates without motion if the gap is closed.
    r = {'status': 'command_completed', 'data': {'context': pc}}

if r['status'] == 'command_completed':
    r2, c2, s2 = run('place', pc)
    if r2['status'] == 'command_completed':
        placed = bool(s2.get('placed_in_container'))
    elif r2['status'] == 'unknown':
        v, vc, sv = run('verify', c2)
        placed = bool(sv.get('placed_in_container'))

print('placed_flag', placed, flush=True)

obs = get_observation()
print('ee', np.round(obs['robot_cartesian_pos'], 3), flush=True)

# Code block 7
import numpy as np

# The pudding box is still on the floor, the hand is empty, and the arm is in an odd pose. Go home first.
# The wrist camera failed SAM, so use single_view=True from here on.
# Plan: one option-based pick with a fresh measurement of the moved box.
# If that fails, a manual top-down grasp closing across the box's short horizontal axis.
try:
    goto_home_joint_position(); open_gripper()
except Exception as e:
    print('home err', str(e)[:100])

def S(ctx):
    try: return get_abstract_state(ctx)['abstract_state']
    except Exception as e:
        print('abs err', str(e)[:80]); return {}

def run(name, c):
    try: r = execute_option(name, c)
    except Exception as e:
        r = {'status': 'error', 'termination_reason': str(e), 'data': {'context': c}}
    cc = (r.get('data') or {}).get('context') or c
    s = S(cc)
    print('RES', name, r.get('status'), str(r.get('termination_reason'))[:100], s.get('failure_mode'),
          s.get('wrong_object_suspected'), s.get('placed_in_container'), s.get('placement_reason'), flush=True)
    return r, cc, s

def qmul(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def gap(): return float(get_observation()['robot_cartesian_pos'][7])

placed = False
c = {'object': 'chocolate_pudding_box', 'container': 'basket', 'prefer': 'geo', 'sampler_failed': True, 'single_view': True}
r, pc, s = run('pick', c)
if r['status'] == 'option_terminated' and s.get('wrong_object_suspected'):
    run('set_down', pc)
elif r['status'] in ('command_completed', 'unknown'):
    print('gap after pick', round(gap(), 3))
    if gap() > 0.08:
        r2, c2, s2 = run('place', pc)
        if r2['status'] == 'unknown':
            v, vc, sv = run('verify', c2); placed = bool(sv.get('placed_in_container'))
        else:
            placed = bool(s2.get('placed_in_container'))
    else:
        run('set_down', dict(pc, mark_reject=False))

# Manual fallback
if not placed and r['status'] not in ('command_completed',):
    try:
        goto_home_joint_position(); open_gripper()
        seg = get_object_3d_points_and_masks_from_language('small brown chocolate pudding box', use_multiview=False)
        P = np.asarray(seg['points_3d']); P = P[np.isfinite(P).all(1)]
        P = P[(P[:, 2] > 0.002) & (P[:, 2] < 0.08)]
        print('npts', len(P), 'score', seg.get('agentview_score'))
        ctr = np.median(P, axis=0)
        xy = P[:, :2] - P[:, :2].mean(0)
        w, V = np.linalg.eigh(np.cov(xy.T))
        a = V[:, 0]  # minor (short) axis
        th = np.arctan2(a[1], a[0])
        top = float(np.percentile(P[:, 2], 95)); bot = float(np.percentile(P[:, 2], 3))
        print('ctr', np.round(ctr, 3), 'ext_sd', np.round(np.sqrt(w), 3), 'top', round(top, 3), 'bot', round(bot, 3))
        held = False
        # The finger closing axis is the hand y-axis. With q0=(0,0,1,0), hand y equals world y, so yaw = th - pi/2.
        for yaw in [th - np.pi/2, th]:
            yaw = (yaw + np.pi/2) % np.pi - np.pi/2
            q = qmul(np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)]), np.array([0., 0., 1., 0.]))
            q /= np.linalg.norm(q)
            gz = max(bot + 0.01, (top + bot) / 2)
            gpos = np.array([P[:, 0].mean(), P[:, 1].mean(), gz])
            open_gripper()
            goto_pose(gpos, q, z_approach=0.12)
            close_gripper()
            goto_pose(gpos + np.array([0, 0, 0.2]), q)
            g = gap(); print('manual yaw', round(yaw, 2), 'gap', round(g, 3), flush=True)
            if g > 0.08:
                held = True; break
            open_gripper()
            goto_pose(gpos + np.array([0, 0, 0.2]), q)
        if held:
            bseg = get_object_3d_points_and_masks_from_language('basket', use_multiview=False)
            B = np.asarray(bseg['points_3d']); B = B[np.isfinite(B).all(1)]
            bc = np.median(B, axis=0); btop = float(np.percentile(B[:, 2], 97))
            print('basket ctr', np.round(bc, 3), 'top', round(btop, 3))
            goto_pose(np.array([bc[0], bc[1], btop + 0.2]), q)
            goto_pose(np.array([bc[0], bc[1], btop + 0.07]), q)
            print('gap over basket', round(gap(), 3))
            open_gripper()
            goto_pose(np.array([bc[0], bc[1], btop + 0.25]), q)
        goto_home_joint_position()
        op, _ = get_object_pose('chocolate_pudding_box', use_multiview=False)
        print('obj after', None if op is None else np.round(op, 3))
    except Exception as e:
        print('manual err', str(e)[:150])
        try: goto_home_joint_position()
        except Exception: pass

print('ee', np.round(get_observation()['robot_cartesian_pos'], 3))

# Code block 8
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')