"""Perception for hammer_nail from the supplied RGB-D observations (no simulator internals).

Depth PNGs are uint16 millimetres along the optical axis; camera poses are OpenGL
(looks along -Z, +Y up). All outputs are world-frame metres.
"""
import base64
import io
import json

import numpy as np
from PIL import Image

# ---- tunables (evidence: E1 hammer_nail_exploration_0, see memory/PROGRESS.md) ----
TABLE_Z_RANGE = (0.80, 1.10)        # plausible table heights (task: 0.92-0.97)
BOARD_MIN_RISE = 0.05               # board top is ~0.11 above table (E1)
HAMMER_MIN_RISE = 0.006             # hammer points must be this far above the table
HAMMER_MAX_RISE = 0.22              # standing hammer is ~0.17 tall
BOARD_MARGIN = 0.02                 # exclude points this close to the board footprint
NAIL_MAX_RISE = 0.12                # nail head is ~0.058 above board (E1)
HANDLE_HALF = 0.015                 # handle centre below its top face (worked in E1)


def quat_wxyz_to_mat(q):
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def flange_pose(state):
    """(position, 3x3 rotation) of the flange from the official 23-D state."""
    return np.asarray(state[0:3], float), quat_wxyz_to_mat(state[3:7])


def decode_reply(reply):
    """Split MCP content blocks into (metadata dict, {(camera, kind): png bytes})."""
    content = reply.get("content", []) if isinstance(reply, dict) else []
    meta = {}
    images = []
    for b in content:
        if b.get("type") == "text" and not meta:
            try:
                meta = json.loads(b.get("text", ""))
            except ValueError:
                meta = {}
        elif b.get("type") == "image":
            images.append(b.get("data"))
    blobs = {}
    att = meta.get("attachments") if isinstance(meta, dict) else None
    if att:
        for a in att:
            k = int(a.get("content_index", 0)) - 1   # content index counts the text block
            if 0 <= k < len(images) and images[k]:
                blobs[(a.get("camera"), a.get("kind"))] = base64.b64decode(images[k])
    return meta, blobs


class Obs:
    """One observation with lazily decoded images and world point clouds."""

    def __init__(self, meta, blobs):
        self.meta = meta or {}
        self.blobs = blobs or {}
        self.state = self.meta.get("state") or self.meta.get("end_state")
        self.step_id = self.meta.get("step_id")
        self._cache = {}

    @classmethod
    def from_reply(cls, reply):
        return cls(*decode_reply(reply))

    @classmethod
    def from_files(cls, meta_path, image_paths):
        """Offline: metadata JSON and image files ordered by content_index (1..4)."""
        meta = json.load(open(meta_path))
        blobs = {}
        for a in meta.get("attachments", []):
            blobs[(a["camera"], a["kind"])] = open(image_paths[a["content_index"] - 1], "rb").read()
        return cls(meta, blobs)

    def has(self, cam):
        return (cam, "rgb") in self.blobs and (cam, "depth") in self.blobs

    def rgb(self, cam):
        key = ("rgb", cam)
        if key not in self._cache:
            self._cache[key] = np.asarray(Image.open(io.BytesIO(self.blobs[(cam, "rgb")])).convert("RGB"))
        return self._cache[key]

    def depth(self, cam):
        key = ("depth", cam)
        if key not in self._cache:
            d = np.asarray(Image.open(io.BytesIO(self.blobs[(cam, "depth")]))).astype(np.float64)
            scale = ((self.meta.get("depth") or {}).get(cam) or {}).get("meters_per_unit", 0.001)
            d = d * scale
            d[d <= 0] = np.nan
            self._cache[key] = d
        return self._cache[key]

    def points(self, cam):
        key = ("pts", cam)
        if key not in self._cache:
            d = self.depth(cam)
            K = self.meta["camera_intrinsics"][cam]["matrix"]
            pose = self.meta["camera_poses"][cam]
            H, W = d.shape
            u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
            xc = (u - K[0][2]) * d / K[0][0]
            yc = -(v - K[1][2]) * d / K[1][1]
            zc = -d
            Rm = quat_wxyz_to_mat(pose["quaternion_wxyz"])
            t = np.asarray(pose["position_m"], float)
            P = np.empty((H, W, 3))
            for i in range(3):
                P[..., i] = Rm[i, 0] * xc + Rm[i, 1] * yc + Rm[i, 2] * zc + t[i]
            self._cache[key] = P
        return self._cache[key]

    def colors(self, cam):
        c = self.rgb(cam).astype(np.int32)
        return c[..., 0], c[..., 1], c[..., 2]


# ---------------- colour classes ----------------
def white_mask(r, g, b):
    return (r > 185) & (g > 185) & (b > 185) & (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b) < 25)


def wood_mask(r, g, b):
    """Light tan handle wood (E1: r-b > 40, r > 80)."""
    return (r - b > 40) & (r > 80) & (g > 45)


def grey_mask(r, g, b):
    """Unsaturated metal (hammer head / nail)."""
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    return (mx - mn < 22) & (mx > 45) & (mx < 238)


def brown_mask(r, g, b):
    return (r - b > 15) & (r < 215) & (g < 180)


# ---------------- scene ----------------
def table_height(obs, cam="front"):
    """Mode of the heights of white points: the table plane dominates (E1: 0.954)."""
    P = obs.points(cam)
    r, g, b = obs.colors(cam)
    z = P[..., 2]
    m = white_mask(r, g, b) & np.isfinite(z) & (z > TABLE_Z_RANGE[0]) & (z < TABLE_Z_RANGE[1])
    zs = z[m]
    if zs.size < 500:
        return None
    hist, edges = np.histogram(zs, bins=np.arange(TABLE_Z_RANGE[0], TABLE_Z_RANGE[1] + 0.001, 0.002))
    k = int(np.argmax(hist))
    sel = zs[(zs >= edges[k] - 0.002) & (zs < edges[k + 1] + 0.002)]
    return float(np.median(sel))


def board_info(obs, table_z, cam="front"):
    """Board top height and xy footprint from brown points well above the table."""
    P = obs.points(cam)
    r, g, b = obs.colors(cam)
    z = P[..., 2]
    m = brown_mask(r, g, b) & np.isfinite(z) & (z > table_z + BOARD_MIN_RISE) & (z < table_z + 0.25)
    m &= (np.abs(P[..., 0]) < 0.9) & (np.abs(P[..., 1]) < 0.9)
    zs = z[m]
    if zs.size < 500:
        return None
    hist, edges = np.histogram(zs, bins=np.arange(table_z + BOARD_MIN_RISE, table_z + 0.25, 0.002))
    k = int(np.argmax(hist))
    top = float(np.median(zs[(zs >= edges[k] - 0.002) & (zs < edges[k + 1] + 0.002)]))
    tp = P[m & (np.abs(z - top) < 0.004)]
    if len(tp) < 300:
        return None
    x0, x1 = np.percentile(tp[:, 0], [0.5, 99.5])
    y0, y1 = np.percentile(tp[:, 1], [0.5, 99.5])
    return {"top": top, "x": [float(x0), float(x1)], "y": [float(y0), float(y1)], "n": int(len(tp))}


def in_box(P, box, margin):
    return ((P[..., 0] > box["x"][0] - margin) & (P[..., 0] < box["x"][1] + margin) &
            (P[..., 1] > box["y"][0] - margin) & (P[..., 1] < box["y"][1] + margin))


def nail_points(obs, cam, board, exclude_above=None):
    """Points standing on the board (nail) within its footprint; optionally below a height cap."""
    P = obs.points(cam)
    z = P[..., 2]
    m = np.isfinite(z) & in_box(P, board, -0.008) & (z > board["top"] + 0.004) & (z < board["top"] + NAIL_MAX_RISE)
    if exclude_above is not None:
        m &= z < exclude_above
    r, g, b = obs.colors(cam)
    m &= ~(brown_mask(r, g, b) & (r - b > 30))   # board texture, not metal
    return P[m]


def nail_info(obs, board, cams=("wrist", "front"), exclude_above=None):
    """Nail head top (x, y, z_top) from the highest compact cluster above the board."""
    best = None
    for cam in cams:
        if not obs.has(cam):
            continue
        pts = nail_points(obs, cam, board, exclude_above)
        if len(pts) < 15:
            continue
        zmax = np.percentile(pts[:, 2], 99.5)
        top = pts[pts[:, 2] > zmax - 0.006]
        xy = np.median(top[:, :2], axis=0)
        # compactness: nail head is ~3 cm wide; reject clusters spread wider than 6 cm
        spread = np.percentile(np.linalg.norm(top[:, :2] - xy, axis=1), 90)
        if spread > 0.03:
            continue
        # the flat top disk alone is seen whole from either camera: its centre is unbiased
        disk = top[(top[:, 2] > zmax - 0.0025) & (np.linalg.norm(top[:, :2] - xy, axis=1) < 0.03)]
        if len(disk) >= 8:
            lo = np.percentile(disk[:, :2], 3, axis=0)
            hi = np.percentile(disk[:, :2], 97, axis=0)
            xy = (lo + hi) / 2
        cand = {"x": float(xy[0]), "y": float(xy[1]), "z": float(zmax), "n": int(len(pts)),
                "spread": float(spread), "cam": cam}
        if best is None or cand["n"] > best["n"]:
            best = cand
    return best


# ---------------- hammer ----------------
def hammer_points(obs, table_z, board, cams=("front", "wrist"), z_cap=None):
    """Wood (handle) and grey (head) points of the hammer lying on the table."""
    wood, grey = [], []
    for cam in cams:
        if not obs.has(cam):
            continue
        P = obs.points(cam)
        z = P[..., 2]
        m = np.isfinite(z) & (z > table_z + HAMMER_MIN_RISE) & (z < table_z + HAMMER_MAX_RISE)
        m &= (np.abs(P[..., 0]) < 0.85) & (np.abs(P[..., 1]) < 0.85)
        if board is not None:
            m &= ~in_box(P, board, BOARD_MARGIN)
        if z_cap is not None:
            m &= z < z_cap
        r, g, b = obs.colors(cam)
        wood.append(P[m & wood_mask(r, g, b)])
        grey.append(P[m & grey_mask(r, g, b)])
    wood = np.concatenate(wood) if wood else np.zeros((0, 3))
    grey = np.concatenate(grey) if grey else np.zeros((0, 3))
    return wood, grey


def _pca_dir(xy):
    c = xy.mean(0)
    cov = np.cov((xy - c).T)
    w, v = np.linalg.eigh(cov)
    return c, v[:, -1], float(np.sqrt(max(w[-1], 0))), float(np.sqrt(max(w[0], 0)))


def largest_cluster_xy(pts, radius=0.05):
    """Keep points within `radius` (xy) of the densest seed: drops stray matches."""
    if len(pts) < 10:
        return pts
    xy = pts[:, :2]
    sub = xy[:: max(1, len(xy) // 400)]
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
    seed = sub[int(np.argmax((d < radius).sum(1)))]
    # grow: points within 0.25 m of the seed (hammer is ~0.22 long)
    return pts[np.linalg.norm(xy - seed, axis=1) < 0.25]


def fit_handle(wood, grey):
    """Handle axis from wood points: direction d (head -> end, unit xy), top-face line z = z0 + slope*a.

    Returns dict or None. a is measured along d from centroid c (xy).
    """
    if len(wood) < 60:
        return None
    wood = largest_cluster_xy(wood)
    c, d, s_long, s_lat = _pca_dir(wood[:, :2])
    tp = wood
    for _ in range(3):
        a = (wood[:, :2] - c) @ d
        keep = np.zeros(len(wood), bool)
        for lo in np.arange(a.min(), a.max() + 0.01, 0.01):
            s = (a >= lo) & (a < lo + 0.01)
            if s.sum() >= 5:
                zt = np.percentile(wood[s, 2], 97)
                keep |= s & (wood[:, 2] > zt - 0.004)
        if keep.sum() < 30:
            break
        tp = wood[keep]
        c, d, s_long, s_lat = _pca_dir(tp[:, :2])
    a_all = (wood[:, :2] - c) @ d
    a_top = (tp[:, :2] - c) @ d
    A = np.stack([a_top, np.ones_like(a_top)], 1)
    slope, z0 = np.linalg.lstsq(A, tp[:, 2], rcond=None)[0]
    perp = np.array([-d[1], d[0]])
    lat = (tp[:, :2] - c) @ perp
    # head side: grey points near the axis line
    head_sign = 0
    head_c = None
    if len(grey) >= 20:
        ga = (grey[:, :2] - c) @ d
        gl = (grey[:, :2] - c) @ perp
        near = (np.abs(gl) < 0.07) & (np.abs(ga) < 0.30)
        if near.sum() >= 20:
            gmed = float(np.median(ga[near]))
            head_sign = -1 if gmed < 0 else 1
            head_c = grey[near].mean(0)
    how = "grey"
    if head_sign == 0:
        # fallback: the head end is the higher end (handle slopes down to its end)
        head_sign = 1 if slope > 0 else -1
        how = "slope"
    if head_sign > 0:   # make d point head -> end
        d = -d
        slope = -slope
        a_all = -a_all
        a_top = -a_top
        lat = -lat
    a_min = float(np.percentile(a_all, 1.5))
    a_max = float(np.percentile(a_all, 98.5))
    return {"c": c.tolist(), "d": d.tolist(), "z0": float(z0), "slope": float(slope),
            "incline_deg": float(np.degrees(np.arctan(-slope))),
            "a_min": a_min, "a_max": a_max, "length": a_max - a_min,
            "width": float(np.percentile(lat, 98) - np.percentile(lat, 2)),
            "n_wood": int(len(wood)), "n_top": int(len(tp)), "head": how,
            "head_c": None if head_c is None else head_c.tolist(), "n_grey": int(len(grey))}


def handle_point(fit, a):
    """World point on the handle axis (centre line) at axis coordinate a."""
    c = np.asarray(fit["c"])
    d = np.asarray(fit["d"])
    top = fit["z0"] + fit["slope"] * a
    return np.array([c[0] + a * d[0], c[1] + a * d[1], top - HANDLE_HALF])


def coarse_hammer(obs, table_z, board):
    """Centroid of hammer-coloured points on the table (front camera), for the survey pose."""
    wood, grey = hammer_points(obs, table_z, board, cams=("front",))
    pts = np.concatenate([wood, grey]) if len(wood) + len(grey) else np.zeros((0, 3))
    if len(pts) < 40:
        return None
    pts = largest_cluster_xy(pts)
    return {"c": pts.mean(0).tolist(), "n_wood": int(len(wood)), "n_grey": int(len(grey))}


# ---------------- held hammer ----------------
def head_points_in_flange(obs, cams=("wrist", "front"), radius=0.32, below=0.0):
    """Grey points near the flange, expressed in the flange frame (z_f down when palm down)."""
    fp, Rf = flange_pose(obs.state)
    out = []
    for cam in cams:
        if not obs.has(cam):
            continue
        P = obs.points(cam)
        r, g, b = obs.colors(cam)
        m = grey_mask(r, g, b) & np.isfinite(P[..., 2])
        m &= np.linalg.norm(P - fp, axis=-1) < radius
        pts = P[m]
        if len(pts) == 0:
            continue
        L = (pts - fp) @ Rf          # world -> flange frame (row vectors)
        L = L[L[:, 2] > below]
        out.append(L)
    return np.concatenate(out) if out else np.zeros((0, 3))


def face_in_flange(L):
    """Striking-face centre (flange frame) from head points: lowest slab (largest z_f)."""
    if len(L) < 40:
        return None
    zf = L[:, 2]
    zmax = np.percentile(zf, 99)
    slab = L[zf > zmax - 0.012]
    if len(slab) < 10:
        return None
    xy = np.median(slab[:, :2], axis=0)
    return {"s": [float(xy[0]), float(xy[1]), float(zmax)], "n": int(len(L)), "n_slab": int(len(slab))}
