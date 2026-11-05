"""Development: rebuild a resume dict from controller_log.txt (last SCENE / SURVEY fit / GRASP plan)
and run the controller from a given state.

usage: resume_ctrl.py <start_state> [stop_after] [lift_dz]
"""
import json
import math
import os
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.insert(0, '.')
sys.path.insert(0, 'code/hammer')
import numpy as np
from api.runtime import Context
from controller import main, CLOSE2
from motion import grasp_rotation, continuous_rotvec, row

start = sys.argv[1]
stop = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '-' else None
lift_dz = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15

if os.path.exists('output/controller_state.json') and '--log' not in sys.argv:
    resume = json.load(open('output/controller_state.json'))
    print('resume from controller_state.json saved at step', resume.get('step'), 'state', resume.get('state'))
else:
    scene = fit = plan = None
    for line in open('output/controller_log.txt'):
        if '] SCENE {' in line:
            scene = json.loads(line.split('] SCENE ', 1)[1])
        elif '] SURVEY fit {' in line:
            f = json.loads(line.split('] SURVEY fit ', 1)[1])
            if f.get('ok'):
                fit = f['fit']
        elif '] GRASP plan {' in line:
            plan = json.loads(line.split('] GRASP plan ', 1)[1])
    rot = grasp_rotation(fit['d'], math.radians(fit['incline_deg']))
    F = np.asarray(plan['F'])
    tgt = row(F + np.array([0, 0, lift_dz]), continuous_rotvec(rot, np.array([-np.pi, 0, 0])), CLOSE2)
    resume = {'state': start, 'scene': {k: scene[k] for k in ('table_z', 'board', 'nail', 'coarse', 'coarse_fit')},
              'fit': fit,
              'grasp': {'g': plan['g'], 'F': plan['F'], 'rot': rot.as_rotvec().tolist(), 'side': plan['side'],
                        'yaw': math.radians(plan['yaw_deg']), 'a_g': None},
              'face_f': None, 'cmd': None, 'target': tgt}
resume['state'] = start
with Context() as session:
    main(session, stop_after=stop, resume=resume)
