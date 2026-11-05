"""Perception helpers for threading: measure needle shaft and ring hole in robot0's wrist camera
(robot0 holding the ring top-down with hand x along the ring axis)."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from geom import deproject, cam_mats


def hand_points(obs, cams, cam='robot0_eye_in_hand', arm=0):
    """Deproject a camera into the given arm's hand frame. Returns (Ph HxWx3, P world HxWx3)."""
    K, T = cam_mats(obs, cam)
    P = deproject(cams[cam + '_depth_m'], K, T)
    e = np.array(obs['robot'][f'robot{arm}_eef_pos'])
    R = Rot.from_quat(obs['robot'][f'robot{arm}_eef_quat']).as_matrix()
    return (P - e) @ R, P


def measure_shaft(Ph, ring_face_x=0.0, half=0.005, ylim=0.012, zlim=(-0.03, 0.03)):
    """Needle shaft seen from above in robot0 hand frame (hand x = ring axis toward the needle, y lateral,
    z down). Shaft points: hand x > ring_face_x + 1 mm, |y| < ylim, z in zlim.
    Returns dict(tip_x, y_c, z_axis(x) line fit: z = z0 + s*x, n)."""
    x, y, z = Ph[..., 0], Ph[..., 1], Ph[..., 2]
    m = (x > ring_face_x + 0.001) & (np.abs(y) < ylim) & (z > zlim[0]) & (z < zlim[1])
    if m.sum() < 20:
        return None
    p = Ph[m]
    # top surface: per x-bin the min z (closest to camera = highest)
    xs = np.arange(p[:, 0].min(), p[:, 0].max(), 0.002)
    rows = []
    for x0 in xs:
        s = p[(p[:, 0] >= x0) & (p[:, 0] < x0 + 0.002)]
        if len(s) < 4:
            continue
        top = s[:, 2].min()
        st = s[s[:, 2] < top + 0.0008]
        rows.append((x0 + 0.001, top, st[:, 1].min(), st[:, 1].max()))
    rows = np.array(rows)
    tip_x = p[:, 0].min()
    use = rows[rows[:, 0] > tip_x + 0.004]  # skip the end face
    A = np.stack([np.ones(len(use)), use[:, 0]], 1)
    z0, s = np.linalg.lstsq(A, use[:, 1] + half, rcond=None)[0]
    y_c = np.median((use[:, 2] + use[:, 3]) / 2)
    width = np.median(use[:, 3] - use[:, 2])
    return dict(tip_x=tip_x, y_c=y_c, z0=z0, slope=s, width=width, n=int(m.sum()), rows=rows)


def measure_ring(Ph):
    """Ring geometry in robot0 hand frame from the wrist view (front face plane and hole bottom edge)."""
    x, y, z = Ph[..., 0], Ph[..., 1], Ph[..., 2]
    # front face: points with |y|<0.015, z in [-0.012, 0.025], x in [0, 0.012]; take mode of x
    m = (np.abs(y) < 0.015) & (z > -0.012) & (z < 0.012) & (x > -0.002) & (x < 0.012)
    p = Ph[m]
    face_x = np.median(p[:, 0]) if len(p) else np.nan
    # hole bottom: near-horizontal surface seen through the hole (x < face_x, z in [0.003, 0.012])
    mb = (np.abs(y) < 0.006) & (x < face_x - 0.0005) & (x > face_x - 0.012) & (z > 0.002) & (z < 0.013)
    pb = Ph[mb]
    bottom_z = np.median(pb[:, 2]) if len(pb) else np.nan
    return dict(face_x=face_x, bottom_z=bottom_z, n_face=len(p), n_bot=len(pb))
