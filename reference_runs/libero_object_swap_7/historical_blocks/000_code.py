# Code block 0
import numpy as np
import math

# Step 1: pick the milk with segmentation/sampler error handling
c={'object':'milk_carton','container':'basket'}
try:
    r=execute_option('pick',c)
except Exception as e:
    r={'status':'error','termination_reason':str(e),'data':{'context':c}}
t=str(r.get('termination_reason'))
print('pick1',r['status'],t[:150])
if r['status']=='error' and ('SAM' in t or 'segment' in t or 'Molmo' in t):
    c={'object':'red_milk_box','container':'basket','single_view':True}
    try:
        r=execute_option('pick',c)
    except Exception as e:
        r={'status':'error','termination_reason':str(e),'data':{'context':c}}
    print('pick1b',r['status'],str(r.get('termination_reason'))[:150])
elif r['status']=='error':
    c={'object':'milk_carton','container':'basket','sampler_failed':True}
    try:
        r=execute_option('pick',c)
    except Exception as e:
        r={'status':'error','termination_reason':str(e),'data':{'context':c}}
    print('pick1c',r['status'],str(r.get('termination_reason'))[:150])

pc=r.get('data',{}).get('context',c)

# Step 2: on handoff, re-pick once with geometric preference
if r['status']=='handoff':
    c2={'object':pc.get('object','milk_carton'),'container':'basket','prefer':'geo'}
    for k in ('container_points','container_pose','single_view'):
        if pc.get(k) is not None:
            c2[k]=pc[k]
    try:
        r=execute_option('pick',c2)
    except Exception as e:
        r={'status':'error','termination_reason':str(e),'data':{'context':c2}}
    pc=r.get('data',{}).get('context',c2)
    print('pick2',r['status'],str(r.get('termination_reason'))[:150])

try:
    s=get_abstract_state(pc)['abstract_state']
    print('pick_state',{k:s.get(k) for k in ('holding','attached','gap','grasp_assessment') if k in s})
except Exception as e:
    print('state err',str(e)[:100])
try:
    print('evidence',str(r.get('data',{}).get('model_evidence'))[:300])
except Exception:
    pass

# Step 3: place if pick completed; for unknown, leave for image inspection next turn
if r['status']=='command_completed':
    try:
        r2=execute_option('place',pc)
    except Exception as e:
        r2={'status':'error','termination_reason':str(e),'data':{'context':pc}}
    print('place',r2['status'],str(r2.get('termination_reason'))[:150])
    try:
        s2=get_abstract_state(r2['data']['context'])['abstract_state']
        print('place_state',s2.get('placed_in_container'),s2.get('placement_reason'),s2.get('carry_xy_err'))
    except Exception as e:
        print('state err',str(e)[:100])
    if r2['status']=='unknown':
        try:
            v=execute_option('verify',r2['data']['context'])
            sv=get_abstract_state(v['data']['context'])['abstract_state']
            print('verify',v['status'],sv.get('placed_in_container'),sv.get('placement_reason'))
        except Exception as e:
            print('verify err',str(e)[:150])
elif r['status']=='unknown':
    print('pick unknown: inspect images next turn to decide place vs set_down')

# Code block 1
print('Decision format invalid or task not verified. Inspect the current images and choose a concrete recovery; FINISH only for observed goal completion.')

# Code block 2
import numpy as np
import math

# No pick/place/verify status was recorded in stdout, so placement is unconfirmed.
# Images show a milk carton in the basket and the arm home and empty.
# Run a motion-free verify (low perception cost) to confirm before finishing.
c={'object':'milk_carton','container':'basket'}
try:
    v=execute_option('verify',c)
except Exception as e:
    v={'status':'error','termination_reason':str(e),'data':{'context':c}}
t=str(v.get('termination_reason'))
print('verify1',v.get('status'),t[:150])
if v.get('status')=='error' and ('SAM' in t or 'segment' in t or 'Molmo' in t):
    c={'object':'red_milk_box','container':'basket','single_view':True}
    try:
        v=execute_option('verify',c)
    except Exception as e:
        v={'status':'error','termination_reason':str(e),'data':{'context':c}}
    print('verify2',v.get('status'),str(v.get('termination_reason'))[:150])
try:
    sv=get_abstract_state(v.get('data',{}).get('context',c))['abstract_state']
    print('verify_state',sv.get('placed_in_container'),sv.get('placement_reason'))
except Exception as e:
    print('state err',str(e)[:120])
try:
    print('evidence',str(v.get('data',{}).get('model_evidence'))[:300])
except Exception:
    pass

# Code block 3
import numpy as np
import math

# The images show the red Milk carton inside the basket, but verify reported unknown (edge_ambiguous).
# Do one bounded re-verify with a different plain name, single view (main camera), and a direct pose comparison.
c={'object':'red_milk_box','container':'basket','single_view':True}
try:
    v=execute_option('verify',c)
except Exception as e:
    v={'status':'error','termination_reason':str(e),'data':{'context':c}}
print('verify2',v.get('status'),str(v.get('termination_reason'))[:150])
vc=v.get('data',{}).get('context',c)
try:
    sv=get_abstract_state(vc)['abstract_state']
    print('verify2_state',sv.get('placed_in_container'),sv.get('placement_reason'),sv.get('object_center'),sv.get('object_bottom'))
except Exception as e:
    print('state err',str(e)[:120])

# Direct geometric check from the main camera: compare the milk position with the basket's extent
try:
    bp=get_oriented_bounding_box_from_3d_points(np.asarray(get_object_3d_points_and_masks_from_language('basket',use_multiview=False)['points_3d'],dtype=np.float64))
    print('basket center',np.round(bp['center'],3),'extent',np.round(bp['extent'],3))
except Exception as e:
    print('basket err',str(e)[:120])
try:
    mp,_=get_object_pose('red_milk_carton',use_multiview=False)
    print('milk pos',None if mp is None else np.round(mp,3))
except Exception as e:
    print('milk err',str(e)[:120])