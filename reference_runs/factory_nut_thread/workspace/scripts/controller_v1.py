"""Frozen feedback controller for Isaac-Factory-NutThread-Direct-v0 (M16 nut onto bolt).

Sensors only: robot proprioception + calibrated RGB-D. The bolt axis/tip is fitted from depth point
clouds at the start (the bolt is a fixed asset), the nut offset in the gripper from the wrist camera.
Then: align nut over bolt axis -> descend to contact -> press lightly while the native executor turns
the hand clockwise (yaw is not controllable except a small rate boost), keeping xy compliant on the axis.
"""
import json
import math
import os
import time

import numpy as np

import robot_api as robot

POS_SCALE = 0.02          # m of target offset per unit action (documented)
ROT_SCALE = 0.097         # rad per unit action (documented)
HORIZON = 450

CFG = dict(
    approach_clear=0.004,     # nut bottom height above bolt tip during xy alignment (m)
    xy_tol=0.0004,            # alignment tolerance before descending (m)
    align_max_steps=60,
    k_xy=0.5,                 # smoothed action per (error/POS_SCALE)
    k_z=0.5,
    press_az=-0.35,           # z action while threading (force ~ 2N*az)
    descend_az=-0.15,
    xy_k_thread=0.3,          # xy gain while threaded (soft centring)
    yaw_boost=True,           # use dRx/dRy second-order term to speed up clockwise yaw
    log_dir=None,
    snap_every=0,
)


# ----------------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------------
def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def yaw_of(q):
    R = quat_to_R(q)
    return math.atan2(R[1, 0], R[0, 0])


def native_yaw(q):
    """Yaw as the native success check sees it: [0,2pi) then values above 235 deg wrapped negative."""
    y = yaw_of(q) % (2 * math.pi)
    return y - 2 * math.pi if y > math.radians(235) else y


def backproject(depth, K, T, offset=(0.5, 0.5)):
    """World XYZ per pixel; depth rays pass through (u+offset[0], v+offset[1]) (operator calibration)."""
    depth = depth.astype(np.float64)
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W) + offset[0], np.arange(H) + offset[1])
    valid = np.isfinite(depth) & (depth > 0)
    z = np.where(valid, depth, np.nan)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], -1)
    return pc @ T[:3, :3].T + T[:3, 3]


def _rot_from_rotvec(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def best_tilt_for_yaw(psi, rz=-0.344):
    """dRx,dRy in {-1,1}^2 that make the executor's extracted target yaw most negative (rotation faster)."""
    cur = np.array([[math.cos(psi), math.sin(psi), 0], [math.sin(psi), -math.cos(psi), 0], [0, 0, -1.0]])
    best, arg = 1e9, (0.0, 0.0)
    for rx in (-1.0, 1.0):
        for ry in (-1.0, 1.0):
            R = _rot_from_rotvec(np.array([rx, ry, rz]) * ROT_SCALE) @ cur
            d = math.atan2(R[1, 0], R[0, 0]) - psi
            d = (d + math.pi) % (2 * math.pi) - math.pi
            if d < best:
                best, arg = d, (rx, ry)
    return arg


class Ema:
    """Mirror of the executor's EMA (0.2) on the first five action channels; inverts it so the smoothed
    command equals a desired value whenever the [-1,1] clip allows."""

    def __init__(self):
        self.s = np.zeros(5)

    def command(self, desired):
        desired = np.clip(np.asarray(desired, float), -1, 1)
        u = np.clip((desired - 0.8 * self.s) / 0.2, -1, 1)
        self.s = 0.2 * u + 0.8 * self.s
        return u


# ----------------------------------------------------------------------------------------------
# perception
# ----------------------------------------------------------------------------------------------
def _cam_points(cams, obs, name):
    c = obs['cameras'][name]
    off = c.get('depth_pixel_ray_offset', (0.5, 0.5))
    return backproject(cams[f'{name}_depth_m'], np.array(c['intrinsic_cv']), np.array(c['cam2world_cv']), off)


def _fit_circle(xy, r0=None, iters=30):
    """Robust algebraic+geometric circle fit (centre, radius)."""
    from scipy.optimize import least_squares
    c0 = np.median(xy, 0)
    p0 = np.r_[c0, r0 if r0 else 0.008]

    def res(p):
        return np.linalg.norm(xy - p[:2], axis=1) - p[2]
    r = least_squares(res, p0, loss='soft_l1', f_scale=0.0005)
    return r.x[:2], r.x[2], float(np.sqrt(np.mean(r.fun ** 2)))


def perceive_bolt(obs, cams, eef):
    """Fit the bolt shank axis (x,y) and tip height from closeup + wrist (+scene) depth."""
    ests = []
    for name, w in (('closeup', 1.0), ('wrist', 1.0), ('scene', 0.2)):
        pw = _cam_points(cams, obs, name)
        ok = np.isfinite(pw).all(-1)
        rel = pw - eef
        m = ok & (rel[..., 2] < -0.002) & (np.linalg.norm(rel[..., :2], axis=-1) < 0.05) & (pw[..., 2] > 0.003)
        pts = pw[m]
        if len(pts) < 30:
            continue
        # keep the dominant vertical object: points close (xy) to the median column
        med = np.median(pts[:, :2], 0)
        pts = pts[np.linalg.norm(pts[:, :2] - med, axis=1) < 0.025]
        if len(pts) < 30:
            continue
        ztip = np.percentile(pts[:, 2], 99.5)
        sh = pts[(pts[:, 2] > ztip - 0.022) & (pts[:, 2] < ztip - 0.002)]
        if len(sh) < 20:
            continue
        c, r, rms = _fit_circle(sh[:, :2])
        if not (0.005 < r < 0.011):
            continue
        ests.append(dict(cam=name, c=c, r=r, rms=rms, ztip=ztip, n=len(sh), w=w * min(1.0, len(sh) / 300)))
    if not ests:
        return None
    W = np.array([e['w'] for e in ests])
    c = (np.array([e['c'] for e in ests]) * W[:, None]).sum(0) / W.sum()
    ztip = float((np.array([e['ztip'] for e in ests]) * W).sum() / W.sum())
    return dict(xy=c, ztip=ztip, ests=ests)


def perceive_nut_offset(obs, cams, eef):
    """Nut centre xy offset from EEF and nut top height (relative to EEF) from yellow top-face points."""
    import cv2
    out = []
    for name in ('wrist', 'closeup'):
        rgb = cams[f'{name}_rgb']
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        yellow = (hsv[..., 1] > 100) & (hsv[..., 0] > 15) & (hsv[..., 0] < 40) & (hsv[..., 2] > 60)
        pw = _cam_points(cams, obs, name)
        rel = pw - eef
        ok = yellow & np.isfinite(rel).all(-1) & (np.linalg.norm(rel[..., :2], axis=-1) < 0.03) \
            & (np.abs(rel[..., 2]) < 0.03)
        if ok.sum() < 50:
            continue
        zz = rel[ok][:, 2]
        ztop = np.percentile(zz, 97)
        top = ok & (np.abs(rel[..., 2] - ztop) < 0.0008)
        tp = rel[top]
        if len(tp) < 100:
            continue
        rr = np.linalg.norm(tp[:, :2] - np.median(tp[:, :2], 0), axis=1)
        inner = tp[rr < np.percentile(rr, 15)]
        try:
            c, r, _ = _fit_circle(inner[:, :2], 0.008)
        except Exception:
            continue
        if not (0.004 < r < 0.011):
            c = tp[:, :2].mean(0)
        out.append(dict(cam=name, off=c, ztop=float(ztop), n=len(tp)))
    if not out:
        return dict(off=np.zeros(2), ztop=0.013, ests=[])
    best = out[0]
    return dict(off=np.asarray(best['off']), ztop=best['ztop'], ests=out)


# ----------------------------------------------------------------------------------------------
# control
# ----------------------------------------------------------------------------------------------
class Logger:
    def __init__(self, d):
        self.d = d
        self.f = None
        if d:
            os.makedirs(d, exist_ok=True)
            self.f = open(os.path.join(d, 'log.jsonl'), 'a')

    def __call__(self, **kw):
        if self.f:
            self.f.write(json.dumps(kw, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n')
            self.f.flush()

    def snap(self, tag):
        if not self.d:
            return
        import cv2
        cams = robot.cameras()
        for k in ('closeup_rgb', 'wrist_rgb'):
            cv2.imwrite(os.path.join(self.d, f'{tag}_{k}.png'), cv2.cvtColor(cams[k], cv2.COLOR_RGB2BGR))
        np.save(os.path.join(self.d, f'{tag}_closeup_depth.npy'), cams['closeup_depth_m'])
        np.save(os.path.join(self.d, f'{tag}_wrist_depth.npy'), cams['wrist_depth_m'])


def _eef(obs):
    return np.array(obs['robot']['eef_position_world'], dtype=float)


def main(cfg=None):
    cfg = dict(CFG, **(cfg or {}))
    log = Logger(cfg.get('log_dir'))
    obs = robot.observe()
    if obs.get('terminal'):
        return obs
    eef = _eef(obs)
    cams = robot.cameras()
    bolt = perceive_bolt(obs, cams, eef)
    nut = perceive_nut_offset(obs, cams, eef)
    log(kind='perception', step=obs['native_step'], eef=eef, bolt=bolt, nut=nut)
    if bolt is None:
        bolt = dict(xy=eef[:2].copy(), ztip=eef[2] - 0.015)
    goal_xy = np.asarray(bolt['xy']) - nut['off']      # EEF xy that puts nut centre on bolt axis
    ztip = bolt['ztip']

    ema = Ema()
    phase = 'align'
    phase_t = 0
    last = obs
    snap_every = cfg.get('snap_every') or 0
    while True:
        eef = _eef(last)
        q = last['robot']['eef_quaternion_wxyz']
        psi = yaw_of(q)
        step = last['native_step']
        d = np.zeros(5)                                  # desired smoothed command (dx,dy,dz,dRx,dRy)
        err_xy = goal_xy - eef[:2]
        if phase == 'align':
            zgoal = ztip + cfg['approach_clear']
            d[0:2] = cfg['k_xy'] * err_xy / POS_SCALE
            d[2] = cfg['k_z'] * (zgoal - eef[2]) / POS_SCALE
            if (np.linalg.norm(err_xy) < cfg['xy_tol'] and abs(zgoal - eef[2]) < 0.0007) \
                    or phase_t > cfg['align_max_steps']:
                phase, phase_t = 'descend', 0
        elif phase == 'descend':
            d[0:2] = cfg['k_xy'] * err_xy / POS_SCALE
            d[2] = cfg['descend_az']
            if eef[2] < ztip - 0.0005 or phase_t > 40:
                phase, phase_t = 'thread', 0
        else:
            d[0:2] = cfg['xy_k_thread'] * err_xy / POS_SCALE
            d[2] = cfg['press_az']
        if cfg['yaw_boost']:
            d[3], d[4] = best_tilt_for_yaw(psi)
        u = ema.command(d)
        a = np.r_[u, 1.0]
        r = robot.step(a.tolist())
        obs2 = dict(r['observation'])
        for k in ('terminal', 'native_success', 'final_native_metric'):
            if k in r:
                obs2[k] = r[k]
        log(kind='step', step=step, phase=phase, eef=eef, yaw=psi, nyaw=native_yaw(q), a=a, s=ema.s,
            succ=obs2.get('native_success'), term=obs2.get('terminal'), metric=obs2.get('final_native_metric'),
            tau=obs2['robot']['applied_joint_torques'], qd=obs2['robot']['joint_velocities'])
        phase_t += 1
        last = obs2
        if snap_every and (step + 1) % snap_every == 0:
            log.snap(f's{step + 1:03d}')
        if obs2.get('terminal'):
            log(kind='end', step=obs2['native_step'], succ=obs2.get('native_success'),
                metric=obs2.get('final_native_metric'), eef=_eef(obs2), nyaw=native_yaw(obs2['robot']['eef_quaternion_wxyz']))
            return obs2


if __name__ == '__main__':
    out = main(json.loads(os.environ.get('CTRL_CFG', '{}')))
    print(json.dumps({k: out.get(k) for k in ('native_step', 'terminal', 'native_success', 'final_native_metric')}))
