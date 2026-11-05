"""Dev rollout with injected perception faults to exercise the remaining recovery paths on the real sim:
 (a) agentview 'misses' the needle in the first 3 looks -> main() must move both arms out of the way and look again,
 (b) the first agentview needle estimate has its shaft direction rotated by +90 deg -> robot1 hovers with a wrong grasp
     yaw; the wrist-view shaft votes must flip the direction, re-plan the yaw, re-hover and grasp correctly."""
import sys, json, os, time
sys.path.insert(0, '/workspace')
import numpy as np
import robot_api as robot
tag = sys.argv[1] if len(sys.argv) > 1 else 'faults3'
os.environ['TLIB_LOG'] = f'/workspace/data/steplog_{tag}.jsonl'
os.environ['CTRL_SAVE'] = tag
r = robot.reset()
print('reset ->', {k: v for k, v in r.items() if k not in ('observation', 'robot', 'cameras')})
import tlib
tlib.LOG_PATH = os.environ['TLIB_LOG']
import controller
import perception as pc

state = dict(av_calls=0)
_agentview = pc.agentview_objects
def agentview_faulty(*a, **k):
    est = _agentview(*a, **k)
    state['av_calls'] += 1
    n = state['av_calls']
    if n <= 3 and 'needle' in est:
        est = {kk: v for kk, v in est.items() if kk != 'needle'}
        print(f'FAULT: agentview call {n}: needle dropped')
    elif n == 4 and 'needle' in est:
        nd = dict(est['needle'])
        nd['shaft_yaw'] = float(controller.wrap(nd['shaft_yaw'] + np.pi / 2))
        est = dict(est, needle=nd)
        print('FAULT: agentview call 4: shaft direction rotated +90 deg')
    return est
pc.agentview_objects = agentview_faulty

t0 = time.time()
try:
    controller.main()
finally:
    o = robot.observe()
    print('END step', o['native_step'], 'success', o['native_success'], 'terminal', o['terminal'],
          'dev_remaining', o['development_remaining'], 'wall %.0fs' % (time.time() - t0))
    try:
        np.savez_compressed(f'/workspace/data/{tag}_end.npz', **robot.cameras())
        json.dump(o, open(f'/workspace/data/{tag}_end_obs.json', 'w'))
    except Exception as e:
        print('final grab failed', e)
