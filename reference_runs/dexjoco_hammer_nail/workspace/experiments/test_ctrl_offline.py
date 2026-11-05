"""Offline: run controller perception methods on saved E1 observations."""
import os
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.insert(0, 'code/hammer')
sys.path.insert(0, 'experiments')
import numpy as np
from test_percep_lib import load_run_call, load_saved
import controller as C
from robot import Log


class FakeRobot:
    step_id = 0
    remaining = 1000


c = C.Controller.__new__(C.Controller)
c.log = Log('output/offline_log.txt')
c.rb = FakeRobot()
c.scene = None
c.fit = None
c.grasp = None
c.face_f = None
obs0 = load_run_call(1)
c.perceive_scene(obs0)
for w in sys.argv[1:]:
    obs = load_run_call(int(w[3:])) if w.startswith('run') else load_saved(w)
    fm = c.measure_face(obs)
    print(w, 'face', None if fm is None else {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v) for k, v in fm.items()})
