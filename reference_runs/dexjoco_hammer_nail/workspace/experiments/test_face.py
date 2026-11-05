"""Offline: nail per camera and face measurement on E1 saved observations."""
import os
import sys
import json

os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.insert(0, 'code/hammer')
sys.path.insert(0, 'experiments')
import numpy as np
import perception as pc
from test_percep_lib import load_run_call, load_saved
import controller as C

for w in sys.argv[1:]:
    obs = load_run_call(int(w[3:])) if w.startswith('run') else load_saved(w)
    tz = pc.table_height(obs)
    bd = pc.board_info(obs, tz)
    for cam in ('wrist', 'front'):
        print(w, cam, 'nail', pc.nail_info(obs, bd, cams=(cam,)))
    c = C.Controller.__new__(C.Controller)
    c.scene = {'table_z': tz, 'board': bd, 'nail': {'x': -0.012, 'y': 0.021, 'z': 1.1227}}
    fm = c.measure_face(obs)
    if fm:
        print(w, 'face', {k: np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v for k, v in fm.items()})
    else:
        print(w, 'face None')
