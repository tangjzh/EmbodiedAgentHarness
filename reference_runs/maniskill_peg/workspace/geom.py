"""Camera / pose geometry helpers (numpy only)."""
import numpy as np


def cam_mats(obs, cam):
    c = obs['cameras'][cam]
    K = np.array(c['intrinsic_cv'])[0] if np.array(c['intrinsic_cv']).ndim == 3 else np.array(c['intrinsic_cv'])
    E = np.array(c['extrinsic_cv'])
    E = E[0] if E.ndim == 3 else E
    return K, E


def backproject(depth, K, E):
    """depth (H,W) metres (z-depth) -> world points (H,W,3) using OpenCV extrinsic world->cam."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    # pixel centres are (col + 0.5, row + 0.5) (CAMERA_CONVENTIONS.txt)
    x = (u + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (v + 0.5 - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], -1)
    R, t = E[:, :3], E[:, 3]
    pw = (pc - t) @ R  # R^T (pc - t)
    return pw


def project(pw, K, E):
    R, t = E[:, :3], E[:, 3]
    pc = pw @ R.T + t
    uv = pc[..., :2] / pc[..., 2:3]
    uv = uv * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]]) - 0.5
    return uv, pc[..., 2]  # uv in (col, row) index units


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat_wxyz(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def rotvec(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def rotvec_to_R(v):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(v).as_matrix()
