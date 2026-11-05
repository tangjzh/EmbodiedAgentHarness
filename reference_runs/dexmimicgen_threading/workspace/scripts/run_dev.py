"""Dev rollout: spend one development reset, then run the frozen controller.main() on the fresh episode."""
import sys, json, os, time
sys.path.insert(0, '/workspace')
import numpy as np
import robot_api as robot
tag = sys.argv[1] if len(sys.argv) > 1 else 'dev'
os.environ['TLIB_LOG'] = f'/workspace/data/steplog_{tag}.jsonl'
os.environ['CTRL_SAVE'] = tag
r = robot.reset()
print('reset ->', type(r).__name__, {k: v for k, v in r.items() if k not in ('observation', 'robot', 'cameras')} if isinstance(r, dict) else '')
import tlib
tlib.LOG_PATH = os.environ['TLIB_LOG']
import controller
t0 = time.time()
try:
    controller.main()
finally:
    o = robot.observe()
    print('END step', o['native_step'], 'success', o['native_success'], 'terminal', o['terminal'],
          'dev_remaining', o['development_remaining'], 'wall %.0fs' % (time.time() - t0))
    try:
        cams = robot.cameras()
        np.savez_compressed(f'/workspace/data/{tag}_end.npz', **cams)
        json.dump(o, open(f'/workspace/data/{tag}_end_obs.json', 'w'))
    except Exception as e:
        print('final grab failed', e)
