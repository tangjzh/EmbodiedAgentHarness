"""Grasp planning helpers (development)."""
import numpy as np
from scipy.spatial.transform import Rotation as R

# grasp point (handle axis centre) expressed in the flange frame (palm-down hand)
GRASP_OFFSET_F = np.array([0.085, 0.0, 0.055])


def grasp_frame(handle_dir_xy, incline_deg, toward_end=True):
    """Rotation whose x = fingers (horizontal, perpendicular to handle, yaw chosen so fingers point
    to the right of the handle direction), y = handle direction toward the end (inclined down), z = palm normal (down)."""
    d = np.asarray(handle_dir_xy, float)
    d = d / np.linalg.norm(d)
    x = np.array([-d[1], d[0], 0.0])  # rotate +90 deg
    # choose fingers so that z points down: z = x cross y
    c, s = np.cos(np.radians(incline_deg)), np.sin(np.radians(incline_deg))
    y = np.array([c * d[0], c * d[1], -s])
    z = np.cross(x, y)
    if z[2] > 0:
        x = -x
        z = np.cross(x, y)
    M = np.stack([x, y, z], 1)
    return R.from_matrix(M)


def flange_for_grasp(g_world, rot, offset_f=GRASP_OFFSET_F):
    return np.asarray(g_world) - rot.apply(offset_f)
