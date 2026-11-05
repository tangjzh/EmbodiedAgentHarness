"""Offline helper: crop+upscale rope regions of a capture dir. usage: crops.py dir"""
import sys

import cv2
import numpy as np

d = sys.argv[1].rstrip('/') + '/'
spec = {'scene_right': (60, 160, 120, 230, 7), 'scene_left': (60, 160, 40, 190, 6), 'wrist_right': (40, 200, 30, 300, 3), 'wrist_left': (40, 200, 30, 320, 3)}
for cam, (r0, r1, c0, c1, f) in spec.items():
    rgb = np.load(d + cam + '_rgb.npy')[r0:r1, c0:c1]
    cv2.imwrite(d + cam + '_crop.png', cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC))
print('ok')
