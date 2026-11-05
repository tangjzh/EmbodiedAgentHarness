"""Camera geometry helpers (CV convention: x right, y down, z forward)."""
import numpy as np


def deproject(depth, K, T):
    """depth HxW metres -> world points HxWx3 using intrinsics K and cam2world T."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (u + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (v + 0.5 - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], -1)
    return pc @ T[:3, :3].T + T[:3, 3]


def project(pts, K, T):
    """world points Nx3 -> continuous pixel uv Nx2 (pixel centre at +0.5) and depth N."""
    pts = np.atleast_2d(pts)
    Ti = np.linalg.inv(T)
    pc = pts @ Ti[:3, :3].T + Ti[:3, 3]
    uv = pc[:, :2] / pc[:, 2:3] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
    return uv, pc[:, 2]


def cam_mats(obs, name):
    c = obs['cameras'][name]
    return np.array(c['intrinsic_cv']), np.array(c['cam2world_cv'])
