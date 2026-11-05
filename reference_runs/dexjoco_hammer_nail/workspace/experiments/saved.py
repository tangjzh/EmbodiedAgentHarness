"""Point clouds from a call saved by experiments/call.py (output/calls/<tag>.json + images)."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, 'experiments')
from geom import cam_points

CALLS = Path('output/calls')


def load_saved(tag):
    meta = json.load(open(CALLS / f'{tag}.json'))
    att = meta.get('attachments')
    imgs = {}
    # attachments give content_index (1-based among content blocks; text is index 0)
    files = sorted(CALLS.glob(f'{tag}_img*.png'), key=lambda p: int(p.stem.split('img')[-1]))
    if att:
        for a in att:
            f = CALLS / f"{tag}_img{a['content_index']}.png"
            imgs[(a['camera'], a['kind'])] = f
    else:
        names = [('front', 'rgb'), ('wrist', 'rgb'), ('front', 'depth'), ('wrist', 'depth')]
        for n, f in zip(names, files):
            imgs[n] = f
    return meta, imgs


def cloud(tag, camera):
    meta, imgs = load_saved(tag)
    rgb = np.array(Image.open(imgs[(camera, 'rgb')]).convert('RGB'))
    dep = np.array(Image.open(imgs[(camera, 'depth')])).astype(np.float64) * 0.001
    P = cam_points(dep, meta['camera_intrinsics'][camera]['matrix'], meta['camera_poses'][camera])
    return rgb, dep, P, meta
