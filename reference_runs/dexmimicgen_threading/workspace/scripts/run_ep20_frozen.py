"""Final validation in the evaluation configuration: one development reset, then the frozen controller.main() with
no diagnostic environment variables (no step log, no camera snapshots) and no fault injection."""
import sys, json, os, time
for k in ('TLIB_LOG', 'CTRL_SAVE'):
    os.environ.pop(k, None)
sys.path.insert(0, '/workspace')
import numpy as np
import robot_api as robot
r = robot.reset()
print('reset ->', {k: v for k, v in r.items() if k not in ('observation', 'robot', 'cameras')} if isinstance(r, dict) else r)
import controller
t0 = time.time()
try:
    controller.main()
finally:
    o = robot.observe()
    print('END step', o['native_step'], 'success', o['native_success'], 'terminal', o['terminal'],
          'dev_remaining', o['development_remaining'], 'wall %.0fs' % (time.time() - t0))
    try:   # record the end state for the log (outside main)
        np.savez_compressed('/workspace/data/ep20_end.npz', **robot.cameras())
        json.dump(o, open('/workspace/data/ep20_end_obs.json', 'w'))
    except Exception as e:
        print('final grab failed', e)
