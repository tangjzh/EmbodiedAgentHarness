# Code block 0
# Pick the alphabet soup can and place it in the basket.
# Two cans are visible, so first check which one segmentation picks for "alphabet_soup".
import numpy as np

def info(name):
    try:
        d = get_object_3d_points_and_masks_from_language(name)
        p = d.get('points_3d')
        if p is None or len(p) == 0:
            return None
        obb = get_oriented_bounding_box_from_3d_points(np.asarray(p, dtype=np.float64))
        print(name, 'n', len(p),
              'center', np.round(obb['center'], 3),
              'extent', np.round(obb['extent'], 3),
              'score', d.get('agentview_score'))
        return obb
    except Exception as e:
        print(name, 'seg err', str(e)[:100])
        return None

info('alphabet_soup')
info('alphabet_soup_can')
info('basket')

def ctx_of(r):
    return (r.get('data', {}) or {}).get('context', {}) or {}

def summarize(r, tag):
    pc = ctx_of(r)
    s = get_abstract_state(pc).get('abstract_state', {})
    me = pc.get('model_evidence') or {}
    print(tag, r.get('status'), str(r.get('termination_reason'))[:120],
          'bottom', s.get('object_bottom_source'),
          'h', s.get('object_height'),
          'support', s.get('support_z'),
          'mode', me.get('mode'))
    return pc

# Pick attempt 1: default grasp sampler.
r = execute_option('pick', {'object': 'alphabet_soup', 'container': 'basket'})
pc = summarize(r, 'pick1')

# Sampler error: retry with the sampler-free geometric path.
if r.get('status') == 'error':
    r = execute_option('pick', {'object': 'alphabet_soup', 'container': 'basket',
                                'sampler_failed': True})
    pc = summarize(r, 'pick1b')

# Handoff: no attachment found, so retry with an alternative geometric grasp.
if r.get('status') == 'handoff':
    r = execute_option('pick', {'object': 'alphabet_soup_can', 'container': 'basket',
                                'sampler_failed': True, 'prefer': 'geo_alt'})
    pc = summarize(r, 'pick2')

# Place if the pick completed or is possibly held.
if r.get('status') in ('command_completed', 'unknown'):
    r2 = execute_option('place', pc)
    c2 = ctx_of(r2)
    print('place', r2.get('status'), str(r2.get('termination_reason'))[:120])
    v = execute_option('verify', c2)
    vs = get_abstract_state(ctx_of(v)).get('abstract_state', {})
    print('verify', v.get('status'), vs.get('placed_in_container'), vs.get('placement_reason'))
else:
    print('no place; status', r.get('status'))

# Final gripper state for checking against the images.
obs = get_observation()
print('ee', np.round(obs['robot_cartesian_pos'], 3))