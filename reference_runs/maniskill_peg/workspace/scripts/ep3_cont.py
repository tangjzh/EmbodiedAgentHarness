"""Continue live episode 3 (peg grasped on the table at step 23; hole not found at t=0 because the face is
turned away from the base camera): controller transport() with the active wrist look, then insert()."""
import sys, json
import numpy as np
sys.path.insert(0, '/workspace')
import robot_api as robot
import controller as C

r = C.Runner()
r.L = 0.0985
r.r = 0.0165
r.widths = [0.0329]
r.peg = dict(L_hand=0.0985)
r.holes = []
r.hole = r.xb = r.h = None
r.face = None
r.face_cam = None
r.update_holes(('base_camera',))
print('hole', r.hole, 'xb', r.xb, 'h', r.h, 'L', r.L, 'face', None if r.face is None else np.round(r.face['face_pt'], 4))
try:
    if (r.hole is not None or r.face is not None) and r.transport():
        r.insert()
finally:
    r.info['success'] = r.succ
    r.info['final_step'] = r.n
    r.info['holes'] = r.holes
    r.save_log()
    try:
        cams = robot.cameras()
        np.savez_compressed('/workspace/data/ep3_end.npz', **cams)
    except Exception as e:
        print('cams', e)
    json.dump(r.obs, open('/workspace/data/ep3_end_obs.json', 'w'))
    print('[done] success', r.succ, 'step', r.n)
