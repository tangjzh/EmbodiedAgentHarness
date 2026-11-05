"""Offline perception test on saved observations.

usage: test_percep.py
"""
import os
import sys
import json
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.insert(0, 'code/hammer')
import numpy as np
import perception as pc

RUN = 'runtime/autonomous_controller/runs/interactive-6667dfe599094cffb8d1e7a3f962ff3e'


def load_run_call(n):
    d = f'{RUN}/calls/{n:06d}'
    resp = json.load(open(f'{d}/response.json'))
    meta = json.loads(resp['content'][0]['text'])
    blobs = {}
    for a in meta.get('attachments', []):
        blobs[(a['camera'], a['kind'])] = open(f"{d}/image_{a['content_index']:02d}.png", 'rb').read()
    return pc.Obs(meta, blobs)


def load_saved(tag):
    meta = json.load(open(f'output/calls/{tag}.json'))
    blobs = {}
    for a in meta.get('attachments', []):
        blobs[(a['camera'], a['kind'])] = open(f"output/calls/{tag}_img{a['content_index']}.png", 'rb').read()
    return pc.Obs(meta, blobs)


which = sys.argv[1:] or ['run1', 'step_pregrasp', 'step_carry1', 'step_align1']
for w in which:
    t0 = time.time()
    obs = load_run_call(int(w[3:])) if w.startswith('run') else load_saved(w)
    tz = pc.table_height(obs)
    bd = pc.board_info(obs, tz)
    nl = pc.nail_info(obs, bd)
    print(f'== {w} step {obs.step_id}: table {tz:.4f} board {bd} ')
    print('   nail', nl)
    co = pc.coarse_hammer(obs, tz, bd)
    print('   coarse hammer', co)
    for cams in (('front',), ('wrist',), ('front', 'wrist')):
        fp = np.array(obs.state[:3])
        wood, grey = pc.hammer_points(obs, tz, bd, cams=cams, z_cap=fp[2] - 0.08)
        fit = pc.fit_handle(wood, grey)
        if fit:
            print('  ', cams, {k: (np.round(v, 4).tolist() if isinstance(v, (list, float)) else v) for k, v in fit.items()})
        else:
            print('  ', cams, 'no fit', len(wood), len(grey))
    L = pc.head_points_in_flange(obs, cams=('wrist',))
    print('   face (wrist)', pc.face_in_flange(L))
    L = pc.head_points_in_flange(obs, cams=('front',))
    print('   face (front)', pc.face_in_flange(L))
    print('   time', round(time.time() - t0, 2))
