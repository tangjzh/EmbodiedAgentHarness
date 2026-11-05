"""Development helpers: load a saved observation call and back-project depth to world points."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

RUN_ROOT = Path("runtime/autonomous_controller/runs")


def latest_run():
    runs = sorted(RUN_ROOT.glob("interactive-*"), key=lambda p: p.stat().st_mtime)
    return runs[-1]


def load_call(call_dir):
    call_dir = Path(call_dir)
    resp = json.loads((call_dir / "response.json").read_text())
    return call_dir, resp


def find_obs_meta(resp):
    """Return the metadata dict holding camera_intrinsics etc."""
    if isinstance(resp, dict) and "camera_intrinsics" in resp:
        return resp
    if isinstance(resp, dict) and resp.get("type") == "text":
        try:
            return find_obs_meta(json.loads(resp["text"]))
        except (ValueError, TypeError):
            return None
    if isinstance(resp, dict):
        for v in resp.values():
            r = find_obs_meta(v)
            if r is not None:
                return r
    if isinstance(resp, list):
        for v in resp:
            r = find_obs_meta(v)
            if r is not None:
                return r
    return None


def cam_points(depth_m, K, pose):
    """World points (H,W,3) from linear depth (H,W) with OpenGL camera pose."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    xc = (u - cx) * depth_m / fx
    yc = -(v - cy) * depth_m / fy
    zc = -depth_m
    pc = np.stack([xc, yc, zc], -1)
    q = pose["quaternion_wxyz"]
    rot = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return pc @ rot.T + np.asarray(pose["position_m"])
