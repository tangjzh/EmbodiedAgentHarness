"""Geometry helpers for analysing saved RGB-D dumps offline."""
import json

import numpy as np


def load(tag):
    d = f'/workspace/data/{tag}'
    obs = json.load(open(f'{d}/obs.json'))
    cams = {}
    for name in ('scene', 'closeup', 'wrist'):
        cams[name] = dict(rgb=np.load(f'{d}/{name}_rgb.npy'), depth=np.load(f'{d}/{name}_depth_m.npy'),
                          K=np.array(obs['cameras'][name]['intrinsic_cv']),
                          T=np.array(obs['cameras'][name]['cam2world_cv']))
    return obs, cams


def backproject(cam, offset=(0.5, 0.5)):
    """World XYZ (H,W,3) for every pixel; NaN where depth invalid."""
    depth = cam['depth'].astype(np.float64)
    H, W = depth.shape
    K, T = cam['K'], cam['T']
    u, v = np.meshgrid(np.arange(W) + offset[0], np.arange(H) + offset[1])
    valid = np.isfinite(depth) & (depth > 0)
    z = np.where(valid, depth, np.nan)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], -1)
    return pc @ T[:3, :3].T + T[:3, 3]


def project(cam, pts):
    pts = np.atleast_2d(pts)
    T = cam['T']
    R, t = T[:3, :3], T[:3, 3]
    pc = (pts - t) @ R
    K = cam['K']
    u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
    v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
    return np.stack([u, v, pc[:, 2]], -1)


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def yaw_of(q):
    R = quat_to_R(q)
    return np.arctan2(R[1, 0], R[0, 0])
