"""TwoArmThreading controller (frozen entry point: main()).

Pipeline (all feedback from proprioception + calibrated RGB-D, no object state):
 1. agentview RGB-D -> coarse needle-handle pose + shaft direction, tripod ring centre + axis.
 2. robot0 hovers over the ring, robot1 over the needle handle; wrist depth refines both.
 3. robot0 grasps the ring side bars top-down (hand x || ring axis); robot1 grasps the handle top-down
    (fingers across the shaft). Both lift.
 4. robot0 brings the tripod to a canonical pose (ring axis = world y, hand x = +y so its wrist camera looks at the
    approaching needle); robot1 turns the needle so the shaft points to -y and brings the tip in front of the ring.
 5. robot0's wrist depth measures the shaft line relative to the ring hole; robot1 is corrected (yaw, pitch,
    lateral, height) until aligned, then pushes the shaft through the ring until native success (shaft centre
    within ~1.1 cm of the ring centre).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tlib import Robot, make_action, errors, GRIP_OPEN, GRIP_CLOSE, POS_SCALE, rotvec_err
import perception as pc

# ---------------------------------------------------------------- parameters
BASE_XY = [np.array([-0.56, -0.25]), np.array([-0.56, 0.25])]
J7_C = 0.79                 # j7 ~ J7_C + azimuth(eef from base) - hand yaw  (top-down hand)
J7_LIM = 2.6                # keep |j7| below this (true limit 2.897)
RING_GRASP_DZ = -0.013      # robot0 eef z = ring top - 13 mm (ring centre height)
CUBE_GRASP_DZ = 0.002       # robot1 eef z = handle centre + 2 mm
RING_CANON = np.array([-0.10, -0.05, 1.05])   # robot0 eef while threading (hand yaw +90 deg)
NEEDLE_STAGE = np.array([-0.10, 0.22, 1.10])  # robot1 waiting pose while robot0 positions the tripod
DROOP = np.radians(-2.45)   # prior pitch correction for needle droop in robot1's grasp (tip-low)
TIP_FROM_EEF = 0.14         # handle centre -> shaft tip
SHAFT_AXIS_BELOW_EEF = 0.0025
PRE_GAP = 0.025             # tip this far in front of the ring face before insertion
INSERT_END = 0.075          # final robot1 eef y = hole_y + INSERT_END (shaft centre ~ 4.5 mm past ring centre)
INSERT_RATE = 0.002         # m/step


def log(*a):
    print(*a, flush=True)


def topdown(theta):
    return Rot.from_euler('z', theta).as_matrix() @ np.diag([1., -1., -1.])


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def azimuth(i, pos):
    d = np.asarray(pos)[:2] - BASE_XY[i]
    return float(np.arctan2(d[1], d[0]))


def j7_pred(i, pos, theta):
    return J7_C + azimuth(i, pos) - theta


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_path(th_from, th_to, step=np.radians(50)):
    """Intermediate yaws from th_from to th_to (no wrapping: the caller picked the feasible representative)."""
    n = int(np.ceil(abs(th_to - th_from) / step))
    return [th_from + (th_to - th_from) * (k + 1) / n for k in range(n)] if n > 0 else [th_to]


def representative(theta, period, i, pos, prefer):
    """Among theta + k*period, choose a j7-feasible one closest to `prefer`."""
    cands = [theta + k * period for k in range(-4, 5)]
    ok = [c for c in cands if abs(j7_pred(i, pos, c)) < J7_LIM]
    if not ok:
        ok = sorted(cands, key=lambda c: abs(j7_pred(i, pos, c)))[:1]
    return min(ok, key=lambda c: abs(c - prefer))


# ---------------------------------------------------------------- waypoint executor
def wp(pos, R, grip, **kw):
    d = dict(pos=np.array(pos, float), R=R, grip=grip, vmax=1.0, wmax=0.8, tol_p=0.003, tol_r=0.03,
             hold=0, ki=0.1, timeout=150)
    d.update(kw)
    return d


def run(rb, plans, max_steps=400, label=''):
    """Execute per-arm waypoint lists in parallel. A waypoint completes when within tolerance (then dwells
    `hold` steps) or after `timeout` steps. Integral action only when within 1 cm (anti-windup).
    Returns True when both plans are finished."""
    idx, cnt, dwell = [0, 0], [0, 0], [0, 0]
    integ = [np.zeros(3), np.zeros(3)]
    for k in range(max_steps):
        acts = []
        finished = True
        for i in range(2):
            t = plans[i][idx[i]]
            cur = rb.arm(i)
            ep, er = errors(cur, t)
            reached = ep < t['tol_p'] and er < t['tol_r']
            if reached:
                dwell[i] += 1
            if (reached and dwell[i] > t['hold']) or cnt[i] >= t['timeout']:
                if cnt[i] >= t['timeout'] and not reached:
                    log(f'  [{label}] arm{i} wp{idx[i]} timeout ep={ep:.4f} er={er:.3f}')
                if idx[i] < len(plans[i]) - 1:
                    idx[i] += 1
                    cnt[i], dwell[i] = 0, 0
                    t = plans[i][idx[i]]
                    ep, er = errors(cur, t)
                else:
                    t = dict(t, done=True)
            if not t.get('done'):
                finished = False
            a = make_action(cur, t, 1.0, 1.0, t['vmax'], t['wmax'])
            e = t['pos'] - cur['pos']
            if t['ki'] > 0 and np.linalg.norm(e) < 0.01:
                integ[i] = np.clip(integ[i] + e, -0.02 / t['ki'], 0.02 / t['ki'])
                a[:3] = np.clip(a[:3] + t['ki'] * integ[i] / POS_SCALE, -1, 1)
            else:
                integ[i] *= 0.9
            acts.append(a)
            cnt[i] += 1
        if finished:
            return True
        rb.step(acts[0], acts[1])
        if rb.done or rb.success:
            return False
    log(f'  [{label}] max_steps reached idx={idx}')
    return False


def hold_wp(arm, grip, **kw):
    return wp(arm['pos'], arm['R'], grip, **kw)


# ---------------------------------------------------------------- phases
def perceive_scene(rb):
    cams = rb.cameras()
    est = pc.agentview_objects(rb.obs, cams)
    r, n = est['ring'], est['needle']
    log('agentview: table %.4f ring c=%s top=%.4f axis=%.1fdeg | needle c=%s face=%.1fdeg shaft=%.1fdeg' % (
        est['table_z'], np.round(r['center'], 4), r['top_z'], np.degrees(r['axis_yaw']),
        np.round(n['center'], 4), np.degrees(n['face_yaw']), np.degrees(n['shaft_yaw'])))
    return est


def needle_plan(phi, cube_pos):
    """Choose robot1 grasp yaw and final yaw. Option a: shaft along -hand x, final yaw +90deg.
    Option b: shaft along +hand x, final yaw -90deg. Returns (theta_grasp, theta_final, sign)."""
    best = None
    for sign, th_f in ((-1, np.pi / 2), (1, -np.pi / 2)):
        th_g0 = phi if sign == 1 else phi + np.pi
        for k in range(-2, 3):
            th_g = th_g0 + 2 * np.pi * k
            j_g = j7_pred(1, cube_pos, th_g)
            j_f = j7_pred(1, RING_CANON + [0, 0.13, 0], th_f)
            if abs(j_g) > J7_LIM or abs(j_f) > J7_LIM:
                continue
            cost = abs(th_f - th_g) + 0.3 * max(abs(j_g), abs(j_f))
            if best is None or cost < best[0]:
                best = (cost, th_g, th_f, sign)
    return best[1:]


def grasp_phase(rb, est):
    """Hover, refine with wrist cameras, grasp ring (robot0) and needle (robot1), lift. Returns info dict."""
    ring, ndl, zt = est['ring'], est['needle'], est['table_z']
    a0, a1 = rb.arm(0), rb.arm(1)
    # robot0 yaw: ring axis (period pi), closest to the final +90deg
    rc = ring['center']
    th0 = representative(ring['axis_yaw'], np.pi, 0, rc, np.pi / 2)
    # robot1 yaw from shaft direction; face yaw refines it (period pi/2)
    phi = ndl['shaft_yaw']
    phi = phi + wrap(4 * (ndl['face_yaw'] - phi)) / 4  # snap shaft yaw to the cube face orientation
    cc = ndl['center']
    th1g, th1f, sign = needle_plan(phi, cc)
    log(f'plan: th0={np.degrees(th0):.1f} th1 grasp={np.degrees(th1g):.1f} final={np.degrees(th1f):.1f} sign={sign}')

    # hover poses (rotate in <=50deg steps, rising first)
    p0h = np.array([rc[0], rc[1], ring['top_z'] + 0.09])
    p1h = np.array([cc[0], cc[1], ndl['top_z'] + 0.08])
    plan0, plan1 = [], []
    ths = yaw_path(yaw_of(a0['R']), th0)
    for j, th in enumerate(ths):
        f = (j + 1) / len(ths)
        pos = a0['pos'] + f * (p0h - a0['pos']) + [0, 0, 0.04 * (1 - f)]
        plan0.append(wp(pos, topdown(th), GRIP_OPEN, tol_p=0.02, tol_r=0.15))
    plan0.append(wp(p0h, topdown(th0), GRIP_OPEN, tol_p=0.002, tol_r=0.01, hold=3))
    ths = yaw_path(yaw_of(a1['R']), th1g)
    for j, th in enumerate(ths):
        f = (j + 1) / len(ths)
        pos = a1['pos'] + f * (p1h - a1['pos']) + [0, 0, 0.04 * (1 - f)]
        plan1.append(wp(pos, topdown(th), GRIP_OPEN, tol_p=0.02, tol_r=0.15))
    plan1.append(wp(p1h, topdown(th1g), GRIP_OPEN, tol_p=0.002, tol_r=0.01, hold=3))
    run(rb, [plan0, plan1], 200, 'hover')

    # wrist refinement
    cams = rb.cameras()
    rr = pc.refine_ring(rb.obs, cams, rc)
    if rr is not None and abs(rr['top_z'] - ring['top_z']) < 0.02 and 0.018 < rr['ext'][0] < 0.03:
        th0 = representative(rr['axis_yaw'], np.pi, 0, rr['center'], th0)
        rc, rtop = rr['center'], rr['top_z']
        log(f'ring refined: c={np.round(rc, 4)} top={rtop:.4f} axis={np.degrees(rr["axis_yaw"]):.1f} ext={np.round(rr["ext"], 4)}')
    else:
        rtop = ring['top_z']
        log('ring refine failed', rr)
    rcb = pc.refine_cube(rb.obs, cams, cc, zt)
    if rcb is not None and np.all(np.abs(rcb['ext'] - 0.04) < 0.006):
        th1g = th1g + wrap(4 * (rcb['face_yaw'] - th1g)) / 4
        cc = rcb['center']
        log(f'cube refined: c={np.round(cc, 4)} face={np.degrees(rcb["face_yaw"]):.1f} ext={np.round(rcb["ext"], 4)}')
    else:
        log('cube refine failed', rcb)

    # descend + grasp
    g0 = np.array([rc[0], rc[1], rtop + RING_GRASP_DZ])
    g1 = np.array([cc[0], cc[1], cc[2] + CUBE_GRASP_DZ])
    R0, R1 = topdown(th0), topdown(th1g)
    plan0 = [wp(g0 + [0, 0, 0.035], R0, GRIP_OPEN, tol_p=0.003, tol_r=0.02),
             wp(g0, R0, GRIP_OPEN, vmax=0.3, tol_p=0.0015, tol_r=0.01, hold=2),
             wp(g0, R0, GRIP_CLOSE, tol_p=0.01, tol_r=0.1, hold=14),
             wp(g0 + [0, 0, 0.10], R0, GRIP_CLOSE, vmax=0.3, tol_p=0.004, tol_r=0.03)]
    plan1 = [wp(g1 + [0, 0, 0.04], R1, GRIP_OPEN, tol_p=0.003, tol_r=0.02),
             wp(g1, R1, GRIP_OPEN, vmax=0.5, tol_p=0.0015, tol_r=0.015, hold=2),
             wp(g1, R1, GRIP_CLOSE, tol_p=0.01, tol_r=0.1, hold=14),
             wp(g1 + [0, 0, 0.12], R1, GRIP_CLOSE, vmax=0.5, tol_p=0.004, tol_r=0.03)]
    run(rb, [plan0, plan1], 220, 'grasp')
    q0, q1 = rb.arm(0)['grip'][0], rb.arm(1)['grip'][0]
    log(f'grasped: step {rb.step_count} grip0={q0:.4f} grip1={q1:.4f} p0={np.round(rb.arm(0)["pos"], 4)} p1={np.round(rb.arm(1)["pos"], 4)}')
    return dict(th0=th0, th1g=th1g, th1f=th1f, sign=sign, ok0=q0 > 0.006, ok1=q1 > 0.006)


def bring_together(rb, info):
    """robot0 -> canonical ring pose (yaw +90deg); robot1 -> staging pose with final yaw."""
    a0, a1 = rb.arm(0), rb.arm(1)
    plan0 = []
    ths = yaw_path(yaw_of(a0['R']), np.pi / 2, np.radians(45))
    for j, th in enumerate(ths):
        f = (j + 1) / len(ths)
        plan0.append(wp(a0['pos'] + f * (RING_CANON - a0['pos']), topdown(th), GRIP_CLOSE, vmax=0.3, wmax=0.4,
                        tol_p=0.01, tol_r=0.1))
    plan0.append(wp(RING_CANON, topdown(np.pi / 2), GRIP_CLOSE, vmax=0.3, wmax=0.4, tol_p=0.001, tol_r=0.006,
                    hold=8, timeout=80))
    plan1 = []
    ths = yaw_path(yaw_of(a1['R']), info['th1f'], np.radians(45))
    for j, th in enumerate(ths):
        f = (j + 1) / len(ths)
        plan1.append(wp(a1['pos'] + f * (NEEDLE_STAGE - a1['pos']), topdown(th), GRIP_CLOSE, vmax=0.5, wmax=0.5,
                        tol_p=0.01, tol_r=0.1))
    plan1.append(wp(NEEDLE_STAGE, topdown(info['th1f']), GRIP_CLOSE, vmax=0.5, tol_p=0.003, tol_r=0.02))
    run(rb, [plan0, plan1], 250, 'together')
    log(f'together: step {rb.step_count} p0={np.round(rb.arm(0)["pos"], 4)} yaw0={np.degrees(yaw_of(rb.arm(0)["R"])):.1f} '
        f'p1={np.round(rb.arm(1)["pos"], 4)} yaw1={np.degrees(yaw_of(rb.arm(1)["R"])):.1f}')


def measure(rb):
    cams = rb.cameras()
    Ph, _ = pc.hand_points(rb.obs, cams)
    ring = pc.measure_ring(Ph)
    sh = pc.measure_shaft(Ph, ring['face_x'] if np.isfinite(ring['face_x']) else 0.005, xmax=0.12)
    return ring, sh


def needle_R(info, pitch=DROOP):
    """robot1 orientation: final yaw with droop compensation. The shaft points to -y world in both grasp options,
    so a negative rotation about world x lifts the tip."""
    return Rot.from_euler('x', pitch).as_matrix() @ topdown(info['th1f'])


def align_and_insert(rb, info):
    a0 = rb.arm(0)
    ring, sh = measure(rb)
    e0 = a0['pos']
    hole_z = ring['hole_z'] if np.isfinite(ring['hole_z']) else 0.0005
    face_x = ring['face_x'] if np.isfinite(ring['face_x']) else 0.005
    hole = np.array([e0[0], e0[1], e0[2] - hole_z])
    face_y = e0[1] + face_x
    log(f'ring in hand: face_x={face_x:.4f} bottom_z={ring["bottom_z"]:.4f} -> hole {np.round(hole, 4)}')
    R1 = needle_R(info)
    pre = np.array([hole[0], face_y + PRE_GAP + TIP_FROM_EEF, hole[2] + SHAFT_AXIS_BELOW_EEF])
    h0 = hold_wp(a0, GRIP_CLOSE)
    run(rb, [[h0], [wp(pre + [0, 0.02, 0.02], R1, GRIP_CLOSE, vmax=0.5, wmax=0.5, tol_p=0.004, tol_r=0.03),
                    wp(pre, R1, GRIP_CLOSE, vmax=0.3, tol_p=0.0008, tol_r=0.006, hold=6, timeout=60)]], 150, 'pre')
    tgt = pre.copy()
    tip_y = None
    for it in range(5):
        ring, sh = measure(rb)
        a1 = rb.arm(1)
        if sh is None:
            log('  shaft not found; nudging toward ring')
            tgt = tgt + [0, -0.01, 0]
            run(rb, [[h0], [wp(tgt, R1, GRIP_CLOSE, vmax=0.3, tol_p=0.0008, tol_r=0.006, hold=6, timeout=40)]], 60, 'nudge')
            continue
        xf = face_x
        ey = sh['y0'] + sh['sy'] * xf               # lateral offset at ring face (hand y = world x)
        ez = sh['z0'] + sh['sz'] * xf - hole_z      # >0: shaft below hole centre
        tip_gap = sh['tip_x'] - face_x
        tip_y = e0[1] + sh['tip_x'] - a1['pos'][1]   # tip relative to robot1 eef (world y)
        log(f'  meas{it}: step {rb.step_count} ey={ey * 1000:.2f}mm ez={ez * 1000:.2f}mm sy={sh["sy"]:.4f} sz={sh["sz"]:.4f} '
            f'tip_gap={tip_gap * 1000:.1f}mm width={sh["width"] * 1000:.1f} bins={sh["nbins"]}')
        rot_ok = abs(sh['sy']) < 0.006 and abs(sh['sz']) < 0.006
        if rot_ok and abs(ey) < 0.0004 and abs(ez) < 0.0004:
            break
        L = a1['pos'][1] - face_y
        psi = sh['sy'] if abs(sh['sy']) >= 0.003 else 0.0
        th = sh['sz'] if abs(sh['sz']) >= 0.003 else 0.0
        R1 = Rot.from_euler('z', psi).as_matrix() @ Rot.from_euler('x', th).as_matrix() @ R1
        tgt = a1['pos'] + np.array([-ey - psi * L, 0.0, ez + th * L])
        # keep the tip in front of the ring while aligning
        if tip_gap < 0.01:
            tgt[1] += 0.01 - tip_gap
        run(rb, [[h0], [wp(tgt, R1, GRIP_CLOSE, vmax=0.2, wmax=0.2, tol_p=0.0005, tol_r=0.004, hold=6, timeout=50)]],
            80, f'align{it}')
    # insertion: target moves along -y, PI on x/z
    start = rb.arm(1)['pos'].copy()
    start[[0, 2]] = tgt[[0, 2]]
    y_end = hole[1] + INSERT_END
    if tip_y is not None and -0.155 < tip_y < -0.125:
        # put the shaft centre (tip + 6 cm) 5 mm past the ring centre
        y_end = hole[1] - 0.005 - (tip_y + 0.06)
    log(f'  tip from eef {tip_y}  y_end={y_end:.4f}')
    log(f'insert: from y={start[1]:.4f} to {y_end:.4f} step {rb.step_count}')
    integ = np.zeros(3)
    ty = start[1]
    for k in range(120):
        ty = max(y_end, ty - INSERT_RATE)
        t = dict(pos=np.array([start[0], ty, start[2]]), R=R1, grip=GRIP_CLOSE)
        cur = rb.arm(1)
        a = make_action(cur, t, 1.0, 1.0, 0.5, 0.5)
        e = (t['pos'] - cur['pos']) * [1, 0, 1]
        integ = np.clip(integ + e, -0.2, 0.2)
        a[:3] = np.clip(a[:3] + 0.1 * integ / POS_SCALE, -1, 1)
        rb.step(make_action(rb.arm(0), h0, 1.0, 1.0), a)
        if k % 5 == 0 or rb.success:
            p = rb.arm(1)['pos']
            log(f'  ins {rb.step_count} y={p[1]:.4f} lag={np.round((t["pos"] - p) * 1000, 1)}mm succ={rb.success}')
        if rb.success or rb.done:
            return rb.success
        lag = cur['pos'][1] - ty
        if lag > 0.015:
            log('  insertion stalled (lag %.1f mm)' % (lag * 1000))
            return False
        if ty <= y_end and abs(rb.arm(1)['pos'][1] - y_end) < 0.001:
            break
    log(f'insert end: step {rb.step_count} p1={np.round(rb.arm(1)["pos"], 4)} succ={rb.success}')
    return rb.success


def main():
    rb = Robot()
    log(f'start step {rb.step_count}')
    # let objects settle
    h0, h1 = hold_wp(rb.arm(0), GRIP_OPEN), hold_wp(rb.arm(1), GRIP_OPEN)
    for _ in range(6):
        rb.step(make_action(rb.arm(0), h0, 1.0, 1.0), make_action(rb.arm(1), h1, 1.0, 1.0))
    est = perceive_scene(rb)
    info = grasp_phase(rb, est)
    if rb.success or rb.done:
        return
    if not (info['ok0'] and info['ok1']):
        log('grasp check failed', info)
    bring_together(rb, info)
    if rb.success or rb.done:
        return
    for attempt in range(3):
        if align_and_insert(rb, info):
            log('SUCCESS at step', rb.step_count)
            return
        if rb.done:
            break
        # back off and retry
        a1 = rb.arm(1)
        run(rb, [[hold_wp(rb.arm(0), GRIP_CLOSE)],
                 [wp(a1['pos'] + [0, 0.05, 0], a1['R'], GRIP_CLOSE, vmax=0.3, tol_p=0.003, tol_r=0.02)]], 40, 'backoff')
    log('finished without success at step', rb.step_count)


if __name__ == '__main__':
    main()
