import json, glob, numpy as np
from motion import tcp_pose
from perception import peg_in_hand
for f in sorted(glob.glob('data/runs/ep*_staged.npz')) + sorted(glob.glob('data/runs/ep*_grasped.npz')):
    tag = f[:-4]
    obs = json.load(open(tag + '_obs.json'))
    cams = dict(np.load(f))
    # get L, r from run log
    ep = tag.split('/')[-1].split('_')[0]
    logs = sorted(glob.glob('data/runs/%s_1*.json' % ep))
    L, r = 0.105, 0.02
    try:
        info = json.load(open(logs[-1]))['info']
        L = info.get('L0', L); r = info.get('r0', r)
    except Exception:
        pass
    p, R = tcp_pose(obs)
    ih = peg_in_hand(obs, cams, p, R, r, L)
    if ih is None:
        print(tag, 'None'); continue
    print('%-28s r %.4f n %5d nor %5d nwh %5d theta %5.2f psi %5.2f xb %s head %s' % (tag.split('/')[-1], r, ih['n'], ih['n_orange'], ih['n_white'],
          np.degrees(ih['theta']), np.degrees(ih['psi']), None if ih['xb'] is None else round(ih['xb'], 4),
          None if ih.get('head') is None else np.round(ih['head'], 4)))
