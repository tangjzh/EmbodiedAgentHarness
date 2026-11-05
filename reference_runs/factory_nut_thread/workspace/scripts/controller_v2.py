"""Frozen feedback controller for Isaac-Factory-NutThread-Direct-v0 (M16 nut onto bolt).

Sensors only: robot proprioception + calibrated RGB-D (pixel-ray offsets from the observation metadata).
At the start the bolt axis/tip is fitted from depth point clouds (the bolt is a fixed asset) and the nut
centre offset in the HAND frame from a hexagon fit to the nut's top face (wrist camera). Then:
  approach  - PD servo of the EEF to the pose that puts the nut centre on the bolt axis, nut bottom a few mm
              above the tip (the goal rotates with the hand yaw because the nut is fixed in the hand);
  descend   - lower slowly until the nut drops onto the thread or contact is detected;
  engage    - press while the executor turns the hand clockwise; spiral xy search if the nut does not drop;
  thread    - press down with xy kept soft, until the native episode reports terminal.
Yaw is not commandable (native unidirectional remap); u_yaw=+1 is the fastest clockwise request and the
dRx/dRy channels are used only for their second-order effect on the executor's extracted yaw target.
"""
import json
import math
import os

import numpy as np

import robot_api as robot

POS_SCALE = 0.02          # m of target offset per unit action (documented)
ROT_SCALE = 0.097         # rad per unit action (documented)
HORIZON = 450
NUT_FLATS = 0.024         # nut width across flats (m)

CFG = dict(
    clear=0.005,              # nut-bottom (~EEF) height above bolt tip while aligning xy (m)
    xy_ready=0.0004,          # xy error to start the descent (m)
    v_ready=0.0003,           # speed (m/step) to start the descent
    xy_safe=0.0015,           # above this xy error never command z below tip + safe_h
    safe_h=0.0025,
    kp=80.0,                  # smoothed action per m of position error
    kd=600.0,                 # smoothed action per (m/step) of velocity
    ki=4.0,                   # xy integral (per m*step), clipped
    smax=1.0,
    descend_rate=0.0006,      # m/step of the descending z reference
    descend_lead=0.003,       # z reference never more than this below the EEF (bounds contact force)
    drop_depth=0.0006,        # EEF below tip by this -> nut is around the bolt (on the thread)
    press_s=-0.35,            # z smoothed action while engaged/threading (force ~ 2 N * s)
    press_schedule=None,      # optional [[s, nsteps], ...] cycled in the thread phase (experiments)
    xy_soft=0.25,             # xy gain factor while threading
    rest_before_search=25,    # engaged steps without a drop before the spiral search starts
    spiral_pitch=0.0004,      # radial growth per turn (m)
    spiral_speed=0.00025,     # arc length per step (m)
    spiral_rmax=0.0025,
    yaw_boost=True,
    log_dir='/tmp/nut_controller',
    snap_every=0,
    save_initial=False,
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


def hand_to_world_xy(v, psi):
    """Hand frame (roll=pi, pitch=0, yaw=psi): x=(c,s), y=(s,-c). The map is its own inverse."""
    c, s = math.cos(psi), math.sin(psi)
    return np.array([c * v[0] + s * v[1], s * v[0] - c * v[1]])


world_to_hand_xy = hand_to_world_xy


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


def _fit_circle(xy, r0=None):
    """Robust geometric circle fit (centre, radius, rms)."""
    from scipy.optimize import least_squares
    c0 = np.median(xy, 0)
    p0 = np.r_[c0, r0 if r0 else 0.008]

    def res(p):
        return np.linalg.norm(xy - p[:2], axis=1) - p[2]
    r = least_squares(res, p0, loss='soft_l1', f_scale=0.0005)
    return r.x[:2], r.x[2], float(np.sqrt(np.mean(r.fun ** 2)))


def perceive_bolt(obs, cams, eef):
    """Fit the bolt shank axis (x,y) and tip height from closeup + wrist + scene depth."""
    ests = []
    for name, w in (('closeup', 1.0), ('wrist', 1.0), ('scene', 0.5)):
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


def _hex_fit(T, q=99.7):
    """Centre/orientation of a hexagon from 2D points of its face via the support function: for the flat
    normals the opposite supports are centre +- flats/2. Returns (centre, phi0_deg, widths)."""
    def at(phi0):
        A, b, w = [], [], []
        for k in range(3):
            a = phi0 + k * math.pi / 3
            n = np.array([math.cos(a), math.sin(a)])
            p = T @ n
            hi, lo = np.percentile(p, q), np.percentile(p, 100 - q)
            A.append(n)
            b.append((hi + lo) / 2)
            w.append(hi - lo)
        return np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0], np.array(w)
    best = None
    for deg in np.arange(0.0, 60.0, 1.0):
        c, w = at(math.radians(deg))
        if best is None or w.sum() < best[0]:
            best = (w.sum(), deg)
    for deg in np.arange(best[1] - 1.0, best[1] + 1.0, 0.1):
        c, w = at(math.radians(deg))
        if w.sum() < best[0]:
            best = (w.sum(), deg)
    c, w = at(math.radians(best[1]))
    return c, best[1] % 60.0, w


def perceive_nut(obs, cams, eef, psi):
    """Nut centre offset from the EEF in the HAND frame (fixed while grasped) and nut top height above the
    EEF, from a hexagon fit to the yellow top-face points. Prefers the wrist camera."""
    import cv2
    out = []
    c, s = math.cos(psi), math.sin(psi)
    for name in ('wrist', 'closeup'):
        rgb = cams[f'{name}_rgb']
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        yellow = (hsv[..., 1] > 100) & (hsv[..., 0] > 15) & (hsv[..., 0] < 40) & (hsv[..., 2] > 60)
        pw = _cam_points(cams, obs, name)
        rel = pw - eef
        ok = yellow & np.isfinite(rel).all(-1) & (np.linalg.norm(rel[..., :2], axis=-1) < 0.03) \
            & (np.abs(rel[..., 2]) < 0.03)
        if ok.sum() < 100:
            continue
        P = rel[ok]
        ztop = np.percentile(P[:, 2], 97)
        T = P[np.abs(P[:, 2] - ztop) < 0.0005]
        if len(T) < 100:
            continue
        H = np.stack([c * T[:, 0] + s * T[:, 1], s * T[:, 0] - c * T[:, 1]], -1)
        ctr, phi0, w = _hex_fit(H)
        good = bool(np.all(np.abs(w - NUT_FLATS) < 0.0008))
        out.append(dict(cam=name, o_hand=ctr, phi0=phi0, widths=w, ztop=float(ztop), n=len(T), good=good))
    for e in out:
        if e['good']:
            return dict(o_hand=np.asarray(e['o_hand']), ztop=e['ztop'], ests=out, src=e['cam'])
    if out:
        e = min(out, key=lambda e: np.abs(e['widths'] - NUT_FLATS).max())
        return dict(o_hand=np.asarray(e['o_hand']), ztop=e['ztop'], ests=out, src=e['cam'] + '?')
    return dict(o_hand=np.zeros(2), ztop=0.013, ests=[], src='none')


# ----------------------------------------------------------------------------------------------
# control
# ----------------------------------------------------------------------------------------------
class Logger:
    """Diagnostics only; never raises (the frozen workspace is read-only, default dir is /tmp)."""

    def __init__(self, d):
        self.d = d
        self.f = None
        if d:
            try:
                os.makedirs(d, exist_ok=True)
                self.f = open(os.path.join(d, 'log.jsonl'), 'a')
            except Exception:
                self.d, self.f = None, None

    def __call__(self, **kw):
        if self.f:
            try:
                self.f.write(json.dumps(kw, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n')
                self.f.flush()
            except Exception:
                pass

    def snap(self, tag, cams=None, full=False):
        if not self.d:
            return
        try:
            import cv2
            cams = cams if cams is not None else robot.cameras()
            names = ('scene', 'closeup', 'wrist') if full else ('closeup', 'wrist')
            for n in names:
                cv2.imwrite(os.path.join(self.d, f'{tag}_{n}_rgb.png'), cv2.cvtColor(cams[f'{n}_rgb'], cv2.COLOR_RGB2BGR))
                np.save(os.path.join(self.d, f'{tag}_{n}_depth.npy'), cams[f'{n}_depth_m'])
        except Exception:
            pass


def _eef(obs):
    return np.array(obs['robot']['eef_position_world'], dtype=float)


def _spiral(t, pitch, speed):
    """Archimedean spiral offset after t steps at constant arc speed (r = pitch*theta/2pi)."""
    if t <= 0:
        return np.zeros(2)
    a = pitch / (2 * math.pi)
    th = math.sqrt(2 * speed * t / a)          # arc length ~ a*th^2/2
    r = a * th
    return np.array([r * math.cos(th), r * math.sin(th)])


def main(cfg=None):
    cfg = dict(CFG, **(cfg or {}))
    log = Logger(cfg.get('log_dir'))
    obs = robot.observe()
    if obs.get('terminal'):
        return obs
    eef = _eef(obs)
    psi = yaw_of(obs['robot']['eef_quaternion_wxyz'])
    cams = robot.cameras()
    bolt = perceive_bolt(obs, cams, eef)
    nut = perceive_nut(obs, cams, eef, psi)
    log(kind='perception', step=obs['native_step'], eef=eef, psi=psi, bolt=bolt, nut=nut,
        episode=obs.get('episode'), cfg=cfg)
    if cfg.get('save_initial'):
        log.snap('s000', cams, full=True)
        log(kind='obs0', obs=obs)
    if bolt is None:                                   # blind fallback: assume bolt under the nut
        bolt = dict(xy=eef[:2] + hand_to_world_xy(nut['o_hand'], psi), ztip=eef[2] - 0.015)
    axis = np.asarray(bolt['xy'], float)
    tip = float(bolt['ztip'])
    o_hand = np.asarray(nut['o_hand'], float)

    ema = Ema()
    phase, phase_t = 'approach', 0
    integ = np.zeros(2)
    prev = eef.copy()
    zref = None
    stall = 0
    popped = 0
    search_t = 0
    search_off = np.zeros(2)
    sched_i, sched_t = 0, 0
    last = obs
    snap_every = cfg.get('snap_every') or 0
    kp, kd = cfg['kp'], cfg['kd']
    while True:
        eef = _eef(last)
        q = last['robot']['eef_quaternion_wxyz']
        psi = yaw_of(q)
        step = last['native_step']
        v = eef - prev
        prev = eef.copy()
        goal_xy = axis + search_off - hand_to_world_xy(o_hand, psi)   # EEF xy putting nut centre on axis
        err_xy = goal_xy - eef[:2]
        exy = float(np.linalg.norm(err_xy))
        h = eef[2] - tip                                               # nut bottom ~ EEF height above tip
        d = np.zeros(5)
        xy_gain = 1.0

        if phase == 'approach':
            zgoal = tip + cfg['clear']
            d[2] = kp * (zgoal - eef[2]) - kd * v[2]
            if exy > cfg['xy_safe'] and h < cfg['safe_h'] + 0.001:
                d[2] = max(d[2], 0.3)                                  # too low while misaligned: go up
            if exy < cfg['xy_ready'] and np.linalg.norm(v) < cfg['v_ready'] and abs(zgoal - eef[2]) < 0.0015:
                phase, phase_t, zref = 'descend', 0, eef[2]
            elif phase_t > 90:                                         # do not stall forever
                phase, phase_t, zref = 'descend', 0, eef[2]
        elif phase == 'descend':
            zref = max(min(zref - cfg['descend_rate'], eef[2]), eef[2] - cfg['descend_lead'], tip - 0.006)
            d[2] = kp * (zref - eef[2]) - kd * v[2]
            if exy > cfg['xy_safe'] and h < cfg['safe_h']:
                d[2] = max(d[2], 0.0)
            stall = stall + 1 if (v[2] > -0.0001 and zref < eef[2] - 0.001) else 0
            if h < -cfg['drop_depth']:
                phase, phase_t = 'thread', 0
            elif stall >= 4:
                phase, phase_t = 'engage', 0
        elif phase == 'engage':
            d[2] = cfg['press_s']
            if h < -cfg['drop_depth']:
                phase, phase_t = 'thread', 0
                axis = axis + search_off                               # the found offset is the better axis
                search_off = np.zeros(2)
            elif phase_t > cfg['rest_before_search']:
                search_t += 1
                search_off = _spiral(search_t, cfg['spiral_pitch'], cfg['spiral_speed'])
                if np.linalg.norm(search_off) > cfg['spiral_rmax']:
                    search_t, search_off = 0, np.zeros(2)              # restart the spiral
                    phase, phase_t, zref = 'retry', 0, eef[2]
        elif phase == 'retry':                                         # lift a little, re-centre, descend again
            d[2] = kp * (tip + 0.002 - eef[2]) - kd * v[2]
            if phase_t > 15:
                phase, phase_t, zref = 'descend', 0, eef[2]
        else:                                                          # thread
            xy_gain = cfg['xy_soft']
            popped = popped + 1 if h > -0.0002 else 0
            if popped > 10:                                            # nut came off the thread: re-engage
                phase, phase_t, popped = 'engage', 0, 0
            sched = cfg.get('press_schedule')
            if sched:
                d[2] = sched[sched_i % len(sched)][0]
                sched_t += 1
                if sched_t >= sched[sched_i % len(sched)][1]:
                    sched_i, sched_t = sched_i + 1, 0
            else:
                d[2] = cfg['press_s']

        if phase in ('approach', 'descend', 'retry'):
            integ = np.clip(integ + err_xy, -0.004, 0.004)
        d[0:2] = xy_gain * (kp * err_xy - kd * v[:2] + (cfg['ki'] * integ if phase != 'thread' else 0))
        lim = cfg['smax']
        n = np.linalg.norm(d[0:2])
        if n > lim:
            d[0:2] *= lim / n
        d[2] = float(np.clip(d[2], -lim, lim))
        if cfg['yaw_boost']:
            d[3], d[4] = best_tilt_for_yaw(psi)
        u = ema.command(d)
        a = np.r_[u, 1.0]
        r = robot.step(a.tolist())
        obs2 = dict(r['observation'])
        for k in ('terminal', 'native_success', 'final_native_metric'):
            if k in r:
                obs2[k] = r[k]
        log(kind='step', step=step, phase=phase, t=phase_t, eef=eef, v=v, yaw=psi, nyaw=native_yaw(q), h=h,
            goal=goal_xy, exy=exy, so=search_off, a=a, s=ema.s, succ=obs2.get('native_success'),
            term=obs2.get('terminal'))
        phase_t += 1
        last = obs2
        if snap_every and (step + 1) % snap_every == 0:
            log.snap(f's{step + 1:03d}')
        if obs2.get('terminal'):
            log(kind='end', step=obs2['native_step'], succ=obs2.get('native_success'),
                metric=obs2.get('final_native_metric'), eef=_eef(obs2),
                nyaw=native_yaw(obs2['robot']['eef_quaternion_wxyz']), tip=tip, axis=axis, o_hand=o_hand)
            return obs2


if __name__ == '__main__':
    out = main(json.loads(os.environ.get('CTRL_CFG', '{}')))
    print(json.dumps({k: out.get(k) for k in ('native_step', 'terminal', 'native_success', 'final_native_metric')}))
