"""Dev rollout with injected faults to exercise the recovery paths not yet seen on the real sim:
 (a) first needle-handle grasp shifted 5 cm away from the shaft (robot1 should close on air -> robot1-only retry
     with agentview re-perception while robot0 keeps holding the tripod),
 (b) first insertion attempt with the shaft's lateral measurement biased by 9 mm (tip should hit the ring side bar
     -> stall -> back off -> clean second attempt)."""
import sys, json, os, time
sys.path.insert(0, '/workspace')
import numpy as np
import robot_api as robot
tag = sys.argv[1] if len(sys.argv) > 1 else 'faults2'
os.environ['TLIB_LOG'] = f'/workspace/data/steplog_{tag}.jsonl'
os.environ['CTRL_SAVE'] = tag
r = robot.reset()
print('reset ->', {k: v for k, v in r.items() if k not in ('observation', 'robot', 'cameras')})
import tlib
tlib.LOG_PATH = os.environ['TLIB_LOG']
import controller
import perception as pc

state = dict(cube_calls=0, insert_attempt=None, tg=None)
_refine_cube = pc.refine_cube
def refine_cube_faulty(*a, **k):
    rc = _refine_cube(*a, **k)
    state['cube_calls'] += 1
    if rc is not None and state['cube_calls'] == 1:
        phi = state['tg']['phi']   # shaft direction; shift the other way
        rc = dict(rc, center=rc['center'] - 0.05 * np.array([np.cos(phi), np.sin(phi), 0.0]))
        print('FAULT: needle grasp target shifted 5 cm away from the shaft')
    return rc
pc.refine_cube = refine_cube_faulty

_rag = controller.refine_and_grasp
def rag_spy(rb, tg, *a, **k):
    state['tg'] = tg
    return _rag(rb, tg, *a, **k)
controller.refine_and_grasp = rag_spy

_measure_shaft = pc.measure_shaft
def measure_shaft_faulty(*a, **k):
    sh = _measure_shaft(*a, **k)
    if sh is not None and state['insert_attempt'] == 0:
        sh = dict(sh, y0=sh['y0'] + 0.009)
    return sh
pc.measure_shaft = measure_shaft_faulty

_align = controller.align_and_insert
def align_faulty(rb, tg, attempt=0):
    state['insert_attempt'] = attempt
    if attempt == 0:
        print('FAULT: shaft lateral position biased by 9 mm for this attempt')
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
