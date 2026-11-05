"""Rope perception helpers (RGB-D only): segmentation, back-projection, fused top-down height maps.

All functions take the observation dict from robot.observe() and the camera dict from robot.cameras().
World frame: z up, floor at z = 0.
"""
import cv2
import numpy as np

CAMS = ("scene_left", "scene_right", "wrist_left", "wrist_right")


def rope_mask(rgb):
    """Rope pixels: blue-grey/cyan rope texture. Floor is darker/more saturated blue, robot is grey (sat ~0)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.int16)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 85) & (h < 100) & (s >= 20) & (s < 120) & (v >= 40)


def backproject(obs, cams, name, mask=None):
    """World points (N,3) for pixels in mask (default: all valid depth).
    Depth rays use the per-camera 'depth_pixel_ray_offset' metadata when present (else pixel centres u+0.5, v+0.5)."""
    dep = np.asarray(cams[name + "_depth_m"], np.float64)
    cal = obs["cameras"][name]
    K = np.asarray(cal["intrinsic_cv"], np.float64)
    T = np.asarray(cal["cam2world_cv"], np.float64)
    ou, ov = cal.get("depth_pixel_ray_offset", (0.5, 0.5))
    ok = np.isfinite(dep) & (dep > 0)
    if mask is not None:
        ok &= mask
    v, u = np.nonzero(ok)
    d = dep[v, u]
    X = np.stack([(u + ou - K[0, 2]) / K[0, 0] * d, (v + ov - K[1, 2]) / K[1, 1] * d, d], -1)
    return X @ T[:3, :3].T + T[:3, 3]


def rope_points(obs, cams, names=CAMS, zmax=1.2):
    """Fused rope point cloud from the given cameras, with the camera index of each point."""
    pts, idx = [], []
    for i, n in enumerate(names):
        p = backproject(obs, cams, n, rope_mask(np.asarray(cams[n + "_rgb"])))
        p = p[(p[:, 2] > -0.01) & (p[:, 2] < zmax)]
        pts.append(p)
        idx.append(np.full(len(p), i))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, int)
    return np.concatenate(pts), np.concatenate(idx)


def height_map(pts, x0, x1, y0, y1, res):
    """Max-z top-down map (rows: +x up, cols: +y right). Empty cells = -1."""
    H, W = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    zb = np.full(H * W, -1.0)
    r = ((x1 - pts[:, 0]) / res).astype(int)
    c = ((pts[:, 1] - y0) / res).astype(int)
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    np.maximum.at(zb, r[ok] * W + c[ok], pts[ok, 2])
    return zb.reshape(H, W)


def render_height(zb, zscale=0.06, marks=(), x1=None, y0=None, res=None):
    """Colour-coded image of a height map; marks = [(x, y)] world points drawn as crosses."""
    valid = zb > -0.5
    col = cv2.applyColorMap((np.clip(zb / zscale, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img = np.zeros_like(col)
    img[valid] = col[valid]
    for (x, y) in marks:
        cv2.drawMarker(img, (int((y - y0) / res), int((x1 - x) / res)), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
    return img
