"""Dev rollout with injected faults to exercise recovery paths of the frozen controller:
 (a) first ring grasp shifted 3 cm along the ring axis (should close on air -> retry),
 (b) first insertion attempt with the shaft height measurement biased by 3 mm (tip should hit the ring -> stall
     -> back off -> clean second attempt)."""
import sys, json, os, time
sys.path.insert(0, '/workspace')
import numpy as np
import robot_api as robot
tag = sys.argv[1] if len(sys.argv) > 1 else 'faults'
os.environ['TLIB_LOG'] = f'/workspace/data/steplog_{tag}.jsonl'
os.environ['CTRL_SAVE'] = tag
r = robot.reset()
print('reset ->', {k: v for k, v in r.items() if k not in ('observation', 'robot', 'cameras')})
import tlib
tlib.LOG_PATH = os.environ['TLIB_LOG']
import controller
import perception as pc

state = dict(ring_calls=0, insert_attempt=None)
_refine_ring = pc.refine_ring
def refine_ring_faulty(*a, **k):
    rr = _refine_ring(*a, **k)
    state['ring_calls'] += 1
    if rr is not None and state['ring_calls'] == 1:
        ax = rr['axis_yaw']
        rr = dict(rr, center=rr['center'] + 0.03 * np.array([np.cos(ax), np.sin(ax), 0.0]))
        print('FAULT: ring target shifted 3 cm along the axis')
    return rr
pc.refine_ring = refine_ring_faulty

_measure_shaft = pc.measure_shaft
def measure_shaft_faulty(*a, **k):
    sh = _measure_shaft(*a, **k)
    if sh is not None and state['insert_attempt'] == 0:
        sh = dict(sh, z0=sh['z0'] + 0.0045)
    return sh
pc.measure_shaft = measure_shaft_faulty

_align = controller.align_and_insert
def align_faulty(rb, tg, attempt=0):
    state['insert_attempt'] = attempt
    if attempt == 0:
        print('FAULT: shaft height biased by 4.5 mm for this attempt')
    return _align(rb, tg, attempt)
controller.align_and_insert = align_faulty

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
