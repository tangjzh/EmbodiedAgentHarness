"""Continue live episode 2 (peg already grasped & lifted at step 18): fresh base-camera hole, then
controller transport()+insert()."""
import sys, json
import numpy as np
sys.path.insert(0, '/workspace')
import robot_api as robot
import controller as C

r = C.Runner()
r.L = 0.0972
r.r = 0.0224
r.widths = [0.0448]
r.holes = []
r.hole = r.xb = r.h = None
r.peg = None
r.update_holes(('base_camera',))
print('hole', r.hole, 'xb', r.xb, 'h', r.h, 'n', r.n)
try:
    if r.hole is not None and r.transport():
        r.insert()
finally:
    r.info['success'] = r.succ
    r.info['final_step'] = r.n
    r.save_log()
    cams = robot.cameras()
    np.savez_compressed('/workspace/data/ep2_end.npz', **cams)
    json.dump(r.obs, open('/workspace/data/ep2_end_obs.json', 'w'))
    print('[done] success', r.succ, 'step', r.n)
