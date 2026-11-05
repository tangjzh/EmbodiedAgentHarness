"""TwoArmThreading controller (frozen entry point: main()).

Pipeline (all feedback from proprioception + calibrated RGB-D, no object state):
 1. agentview RGB-D -> coarse needle-handle pose + shaft direction, tripod ring centre + axis.
 2. robot0 hovers over the ring, robot1 over the needle handle; wrist depth refines both (and re-checks the shaft
    direction).
 3. robot0 grasps the ring side bars top-down (hand x || ring axis); robot1 grasps the handle top-down
    (fingers across the shaft). Both lift. An empty gripper triggers a re-perceive + re-grasp for that arm.
 4. robot0 brings the tripod to a canonical pose (ring axis = world y, hand x = +y so its wrist camera looks at the
    approaching needle); robot1 lifts, turns the needle so the shaft points to -y and stages it.
 5. robot0's wrist depth measures the shaft line relative to the ring hole; robot1 is corrected (yaw, pitch,
    lateral, height) until aligned, then pushes the shaft through the ring until native success (shaft centre
    within ~1.1 cm of the ring centre). Stalls back off and retry.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tlib import Robot, make_action, errors, GRIP_OPEN, GRIP_CLOSE, POS_SCALE
import perception as pc

# ---------------------------------------------------------------- parameters
BASE_XY = [np.array([-0.56, -0.25]), np.array([-0.56, 0.25])]
J7_C = 0.79                 # j7 ~ J7_C + azimuth(eef from base) - hand yaw  (top-down hand)
J7_LIM = 2.6                # keep |j7| below this (true limit 2.897)
RING_GRASP_DZ = -0.013      # robot0 eef z = ring top - 13 mm (ring centre height)
CUBE_GRASP_DZ = 0.002       # robot1 eef z = handle centre + 2 mm
RING_HOVER = 0.09           # robot0 hover height above the ring top
CUBE_HOVER = 0.08           # robot1 hover height above the handle top
GRIP_MIN = 0.006            # finger qpos below this after closing = nothing grasped (ring 0.0119, handle 0.0150)
RING_CANON = np.array([-0.10, -0.05, 1.05])   # robot0 eef while threading (hand yaw +90 deg)
NEEDLE_STAGE = np.array([-0.10, 0.22, 1.10])  # robot1 waiting pose while robot0 positions the tripod
DROOP = np.radians(-1.25)   # prior pitch correction for needle droop in robot1's grasp (tip-low); measured 1.2 deg
TIP_FROM_EEF = 0.14         # handle centre -> shaft tip
SHAFT_AXIS_BELOW_EEF = 0.0025
PRE_GAP = 0.025             # tip this far in front of the ring face before insertion
INSERT_END = 0.075          # default final robot1 eef y = hole_y + INSERT_END (shaft centre ~ at the ring centre)
INSERT_RATE = 0.002         # m/step
SAVE = os.environ.get('CTRL_SAVE')   # dev diagnostics only: save camera frames as data/<SAVE>_<name>.npz


def log(*a):
    print(*a, flush=True)


def deg(a):
    return float(np.round(np.degrees(a), 1))


def snap(rb, name, cams=None):
    if not SAVE:
        return
    try:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        np.savez_compressed(os.path.join(d, f'{SAVE}_{name}.npz'), **(cams if cams is not None else rb.cameras()))
        with open(os.path.join(d, f'{SAVE}_{name}_obs.json'), 'w') as f:
            json.dump(rb.obs, f)
    except Exception as e:
        log('snap failed', repr(e))


def topdown(theta):
    return Rot.from_euler('z', theta).as_matrix() @ np.diag([1., -1., -1.])


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def azimuth(i, pos):
    d = np.asarray(pos)[:2] - BASE_XY[i]
    return float(np.arctan2(d[1], d[0]))


def j7_pred(i, pos, theta):
    return J7_C + azimuth(i, pos) - theta


def cur_yaw(i, arm):
    """Hand yaw unwrapped consistently with joint 7, so yaw paths never wind the wrist into its limit."""
    th = yaw_of(arm['R'])
    est = J7_C + azimuth(i, arm['pos']) - arm['jpos'][6]
    return th + 2 * np.pi * np.round((est - th) / (2 * np.pi))


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


def hold_steps(rb, n, grips):
    h = [hold_wp(rb.arm(i), grips[i]) for i in range(2)]
    for _ in range(n):
        rb.step(make_action(rb.arm(0), h[0], 1.0, 1.0), make_action(rb.arm(1), h[1], 1.0, 1.0))


# ---------------------------------------------------------------- perception -> grasp targets
def perceive_scene(rb, tries=3, grips=(GRIP_OPEN, GRIP_OPEN)):
    """agentview estimate of both objects (retries after holding still if something is missing)."""
    est = None
    for k in range(tries):
        try:
            est = pc.agentview_objects(rb.obs, rb.cameras())
            r, n = est.get('ring'), est.get('needle')
            log('agentview: table %.4f ring %s | needle %s' % (
                est['table_z'],
                'c=%s top=%.4f axis=%.1fdeg' % (np.round(r['center'], 4), r['top_z'], deg(r['axis_yaw'])) if r else None,
                'c=%s top=%.4f face=%.1fdeg shaft=%.1fdeg n_shaft=%d' % (
                    np.round(n['center'], 4), n['top_z'], deg(n['face_yaw']), deg(n['shaft_yaw']), n['n_shaft'])
                if n else None))
            if r is not None and n is not None:
                return est
        except Exception as e:
            log('agentview perception failed:', repr(e))
        hold_steps(rb, 5, grips)
    return est


def needle_dir(n):
    """Shaft yaw snapped to the handle face orientation; if the shaft was not seen, guess the face normal that
    points toward -x (the shaft has pointed toward the robots in every episode so far)."""
    cands = n['face_yaw'] + np.arange(4) * np.pi / 2
    phi = n['shaft_yaw']
    if not np.isfinite(phi) or n.get('n_shaft', 0) < 5:
        phi = np.pi
    return float(wrap(min(cands, key=lambda a: abs(wrap(a - phi)))))


def ring_plausible(r, table_z):
    return r is not None and abs(r['top_z'] - table_z - 0.196) < 0.012


def needle_plausible(n, table_z):
    return n is not None and abs(n['top_z'] - table_z - 0.04) < 0.008 and n.get('n_top', 0) >= 20


def update_targets(tg, est, which=(True, True)):
    tg['table_z'] = est['table_z']
    r, n = est.get('ring'), est.get('needle')
    if which[0] and r is not None:
        tg.update(rc=r['center'], rtop=r['top_z'], raxis=r['axis_yaw'])
    if which[1] and n is not None:
        tg.update(cc=n['center'], ctop=n['top_z'], face=n['face_yaw'], phi=needle_dir(n))
    return tg


def needle_plan(phi, cube_pos):
    """Choose robot1 grasp yaw and final yaw. Option a: shaft along -hand x, final yaw +90deg.
    Option b: shaft along +hand x, final yaw -90deg. Returns (theta_grasp, theta_final, sign)."""
    best, fallback = None, None
    for sign, th_f in ((-1, np.pi / 2), (1, -np.pi / 2)):
        th_g0 = phi if sign == 1 else phi + np.pi
        for k in range(-2, 3):
            th_g = th_g0 + 2 * np.pi * k
            viol = max(abs(j7_pred(1, cube_pos, th_g)), abs(j7_pred(1, RING_CANON + [0, 0.13, 0], th_f)))
            if viol > J7_LIM:
                if fallback is None or viol < fallback[0]:
                    fallback = (viol, th_g, th_f, sign)
                continue
            cost = abs(th_f - th_g) + 0.3 * viol
            if best is None or cost < best[0]:
                best = (cost, th_g, th_f, sign)
    return (best or fallback)[1:]


# ---------------------------------------------------------------- grasping
def hover_plan(start, th_start, p_h, th):
    """Rotate in <=50deg steps while moving (rising first) to the hover pose."""
    plan = []
    ths = yaw_path(th_start, th)
    for j, t in enumerate(ths):
        f = (j + 1) / len(ths)
        pos = start + f * (p_h - start) + [0, 0, 0.04 * (1 - f)]
        plan.append(wp(pos, topdown(t), GRIP_OPEN, tol_p=0.02, tol_r=0.15))
    plan.append(wp(p_h, topdown(th), GRIP_OPEN, tol_p=0.002, tol_r=0.01, hold=3))
    return plan


def hover(rb, tg, active, held, rise=False):
    """Move the active arms (gripper open) above their targets; the others hold (closed if holding)."""
    a = [rb.arm(0), rb.arm(1)]
    plans = [[hold_wp(a[i], GRIP_CLOSE if held[i] else GRIP_OPEN)] for i in range(2)]
    if held[1] and not active[1]:   # lift the held needle out of agentview's object height band
        plans[1] = [wp([*a[1]['pos'][:2], max(a[1]['pos'][2], NEEDLE_STAGE[2])], a[1]['R'], GRIP_CLOSE, vmax=0.5,
                       tol_p=0.01, tol_r=0.05)]
    for i in range(2):
        if not active[i]:
            continue
        if i == 0:
            tg['th0'] = representative(tg['raxis'], np.pi, 0, tg['rc'], tg.get('th0', np.pi / 2))
            p_h, th = np.array([tg['rc'][0], tg['rc'][1], tg['rtop'] + RING_HOVER]), tg['th0']
        else:
            tg['th1g'], tg['th1f'], tg['sign'] = needle_plan(tg['phi'], tg['cc'])
            p_h, th = np.array([tg['cc'][0], tg['cc'][1], tg['ctop'] + CUBE_HOVER]), tg['th1g']
        start, plans[i] = a[i]['pos'], []
        if rise:   # lift straight up first (the arm may be next to its object after a failed grasp)
            start = start + [0, 0, 0.06]
            plans[i].append(wp(start, a[i]['R'], GRIP_OPEN, tol_p=0.01, tol_r=0.1, timeout=40))
        plans[i] += hover_plan(start, cur_yaw(i, a[i]), p_h, th)
    log(f'hover plan: th0={deg(tg.get("th0", np.nan))} th1 grasp={deg(tg.get("th1g", np.nan))} '
        f'final={deg(tg.get("th1f", np.nan))} sign={tg.get("sign")} active={active}')
    run(rb, plans, 220, 'hover')


def refine_and_grasp(rb, tg, active, held, win=1.0):
    """Wrist-camera refinement of the targets, then descend, close and lift with the active arms.
    Returns per-arm 'holding' flags."""
    cams = rb.cameras()
    snap(rb, 'hover%d%d' % tuple(int(x) for x in active), cams)
    a = [rb.arm(0), rb.arm(1)]
    plans = [[hold_wp(a[i], GRIP_CLOSE if held[i] else GRIP_OPEN)] for i in range(2)]
    if active[0]:
        rr = pc.refine_ring(rb.obs, cams, tg['rc'], win=0.04 * win)
        if rr is not None and abs(rr['top_z'] - tg['rtop']) < 0.02 and 0.018 < rr['ext'][0] < 0.03:
            tg.update(rc=rr['center'], rtop=rr['top_z'], raxis=rr['axis_yaw'])
            tg['th0'] = representative(rr['axis_yaw'], np.pi, 0, rr['center'], tg['th0'])
            log(f'ring refined: c={np.round(rr["center"], 4)} top={rr["top_z"]:.4f} axis={deg(rr["axis_yaw"])} '
                f'ext={np.round(rr["ext"], 4)}')
        else:
            log('ring refine failed', rr)
        g0 = np.array([tg['rc'][0], tg['rc'][1], tg['rtop'] + RING_GRASP_DZ])
        R0 = topdown(tg['th0'])
        plans[0] = [wp(g0 + [0, 0, 0.035], R0, GRIP_OPEN, tol_p=0.003, tol_r=0.02),
                    wp(g0, R0, GRIP_OPEN, vmax=0.3, tol_p=0.0015, tol_r=0.01, hold=2),
                    wp(g0, R0, GRIP_CLOSE, tol_p=0.01, tol_r=0.1, hold=14),
                    wp(g0 + [0, 0, 0.10], R0, GRIP_CLOSE, vmax=0.3, tol_p=0.004, tol_r=0.03)]
    if active[1]:
        rcb = pc.refine_cube(rb.obs, cams, tg['cc'], tg['table_z'], win=0.05 * win)
        if rcb is not None and np.all(np.abs(rcb['ext'] - 0.04) < 0.006):
            tg.update(cc=rcb['center'], face=rcb['face_yaw'])
            log(f'cube refined: c={np.round(rcb["center"], 4)} face={deg(rcb["face_yaw"])} ext={np.round(rcb["ext"], 4)}')
        else:
            log('cube refine failed', rcb)
        # shaft direction from the wrist view (votes along the four face normals)
        yaws, cnt = pc.shaft_votes(pc.cloud(rb.obs, cams, 'robot1_eye_in_hand'), tg['cc'], tg['table_z'], tg['face'])
        k_cur = int(np.argmin([abs(wrap(y - tg['phi'])) for y in yaws]))
        k_best = int(np.argmax(cnt))
        phi_old, th_old = tg['phi'], tg['th1g']
        pre1 = []
        if k_best != k_cur and cnt[k_best] >= 30 and cnt[k_cur] < 5:
            tg['phi'] = float(wrap(yaws[k_best]))
            tg['th1g'], tg['th1f'], tg['sign'] = needle_plan(tg['phi'], tg['cc'])
            log(f'shaft direction corrected by wrist view {deg(phi_old)} -> {deg(tg["phi"])} votes {cnt}; '
                f'grasp yaw {deg(th_old)} -> {deg(tg["th1g"])}')
            p_h = np.array([tg['cc'][0], tg['cc'][1], tg['ctop'] + CUBE_HOVER])
            pre1 = hover_plan(a[1]['pos'], cur_yaw(1, a[1]), p_h, tg['th1g'])
        else:
            tg['phi'] = float(wrap(yaws[k_cur]))
            tg['th1g'] = th_old + wrap(tg['phi'] - phi_old)
            log(f'shaft votes {cnt} (dir {deg(tg["phi"])})')
        g1 = np.array([tg['cc'][0], tg['cc'][1], tg['cc'][2] + CUBE_GRASP_DZ])
        R1 = topdown(tg['th1g'])
        plans[1] = pre1 + [wp(g1 + [0, 0, 0.04], R1, GRIP_OPEN, tol_p=0.003, tol_r=0.02),
                           wp(g1, R1, GRIP_OPEN, vmax=0.5, tol_p=0.0015, tol_r=0.015, hold=2),
                           wp(g1, R1, GRIP_CLOSE, tol_p=0.01, tol_r=0.1, hold=14),
                           wp(g1 + [0, 0, 0.12], R1, GRIP_CLOSE, vmax=0.5, tol_p=0.004, tol_r=0.03)]
    run(rb, plans, 260, 'grasp')
    q = [rb.arm(i)['grip'][0] for i in range(2)]
    ok = [bool(held[i] or (active[i] and q[i] > GRIP_MIN)) for i in range(2)]
    log(f'grasp: step {rb.step_count} grip0={q[0]:.4f} grip1={q[1]:.4f} holding={ok} '
        f'p0={np.round(rb.arm(0)["pos"], 4)} p1={np.round(rb.arm(1)["pos"], 4)}')
    return ok


def acquire(rb, tg, max_retries=3):
    """Grasp both objects; arms that come up empty re-perceive (agentview + wider wrist window) and retry."""
    held = [False, False]
    hover(rb, tg, [True, True], held)
    if rb.done or rb.success:
        return held
    held = refine_and_grasp(rb, tg, [True, True], held)
    for attempt in range(max_retries):
        if all(held) or rb.done or rb.success:
            break
        active = [not h for h in held]
        log(f'grasp retry {attempt + 1} for arms {active}')
        hover(rb, tg, active, held, rise=True)
        est = perceive_scene(rb, tries=1, grips=[GRIP_CLOSE if h else GRIP_OPEN for h in held])
        if est is not None:
            ok_r = active[0] and ring_plausible(est.get('ring'), est['table_z'])
            ok_n = active[1] and needle_plausible(est.get('needle'), est['table_z'])
            update_targets(tg, est, (ok_r, ok_n))
            if active[0] and not ok_r:
                log('ring not plausible in agentview (tripod moved/tipped?) - keeping previous estimate')
            hover(rb, tg, active, held)
        held = refine_and_grasp(rb, tg, active, held, win=1.6)
    return held


# ---------------------------------------------------------------- threading
def bring_together(rb, tg):
    """robot0 -> canonical ring pose (yaw +90deg); robot1 lifts, then turns to the final yaw at the staging pose."""
    a0, a1 = rb.arm(0), rb.arm(1)
    plan0 = []
    ths = yaw_path(cur_yaw(0, a0), np.pi / 2, np.radians(45))
    for j, th in enumerate(ths):
        f = (j + 1) / len(ths)
        plan0.append(wp(a0['pos'] + f * (RING_CANON - a0['pos']), topdown(th), GRIP_CLOSE, vmax=0.3, wmax=0.4,
                        tol_p=0.01, tol_r=0.1))
    plan0.append(wp(RING_CANON, topdown(np.pi / 2), GRIP_CLOSE, vmax=0.3, wmax=0.4, tol_p=0.001, tol_r=0.006,
                    hold=8, timeout=80))
    up = np.array([a1['pos'][0], a1['pos'][1], max(a1['pos'][2], NEEDLE_STAGE[2])])
    plan1 = [wp(up, a1['R'], GRIP_CLOSE, vmax=0.5, tol_p=0.015, tol_r=0.05, timeout=60)]
    ths = yaw_path(cur_yaw(1, a1), tg['th1f'], np.radians(45))
    for j, th in enumerate(ths):
        f = (j + 1) / len(ths)
        plan1.append(wp(up + f * (NEEDLE_STAGE - up), topdown(th), GRIP_CLOSE, vmax=0.5, wmax=0.5,
                        tol_p=0.01, tol_r=0.1))
    plan1.append(wp(NEEDLE_STAGE, topdown(tg['th1f']), GRIP_CLOSE, vmax=0.5, tol_p=0.003, tol_r=0.02))
    run(rb, [plan0, plan1], 260, 'together')
    log(f'together: step {rb.step_count} p0={np.round(rb.arm(0)["pos"], 4)} yaw0={deg(yaw_of(rb.arm(0)["R"]))} '
        f'p1={np.round(rb.arm(1)["pos"], 4)} yaw1={deg(yaw_of(rb.arm(1)["R"]))}')


def measure(rb, name=None):
    cams = rb.cameras()
    if name:
        snap(rb, name, cams)
    Ph, _ = pc.hand_points(rb.obs, cams)
    ring = pc.measure_ring(Ph)
    sh = pc.measure_shaft(Ph, ring['face_x'] if np.isfinite(ring['face_x']) else 0.005, xmax=0.12)
    return ring, sh


def needle_R(tg, pitch=DROOP):
    """robot1 orientation: final yaw with droop compensation. The shaft points to -y world in both grasp options,
    so a negative rotation about world x lifts the tip."""
    return Rot.from_euler('x', pitch).as_matrix() @ topdown(tg['th1f'])


def track(rb, R1, xz, ys, h0, label='ins'):
    """robot1 follows the y targets `ys` (PI on x/z) while robot0 holds h0.
    Returns 'success', 'done', 'stall' or 'end'."""
    integ = np.zeros(3)
    for k, ty in enumerate(ys):
        t = dict(pos=np.array([xz[0], ty, xz[1]]), R=R1, grip=GRIP_CLOSE)
        cur = rb.arm(1)
        a = make_action(cur, t, 1.0, 1.0, 0.5, 0.5)
        e = (t['pos'] - cur['pos']) * [1, 0, 1]
        integ = np.clip(integ + e, -0.2, 0.2)
        a[:3] = np.clip(a[:3] + 0.1 * integ / POS_SCALE, -1, 1)
        rb.step(make_action(rb.arm(0), h0, 1.0, 1.0), a)
        if k % 5 == 0 or rb.success:
            p = rb.arm(1)['pos']
            log(f'  {label} {rb.step_count} y={p[1]:.4f} lag={np.round((t["pos"] - p) * 1000, 1)}mm succ={rb.success}')
        if rb.success:
            return 'success'
        if rb.done:
            return 'done'
        if abs(cur['pos'][1] - ty) > 0.015:
            log(f'  {label} stalled (lag {(cur["pos"][1] - ty) * 1000:.1f} mm)')
            return 'stall'
    return 'end'


def align_and_insert(rb, tg, attempt=0):
    a0 = rb.arm(0)
    ring, sh = measure(rb, f'inhand{attempt}')
    e0 = a0['pos']
    hole_z = ring['hole_z'] if np.isfinite(ring['hole_z']) else 0.0005
    face_x = ring['face_x'] if np.isfinite(ring['face_x']) else 0.005
    hole = np.array([e0[0], e0[1], e0[2] - hole_z])
    face_y = e0[1] + face_x
    log(f'ring in hand: face_x={face_x:.4f} bottom_z={ring["bottom_z"]:.4f} -> hole {np.round(hole, 4)}')
    R1 = needle_R(tg)
    pre = np.array([hole[0], face_y + PRE_GAP + TIP_FROM_EEF, hole[2] + SHAFT_AXIS_BELOW_EEF])
    h0 = hold_wp(a0, GRIP_CLOSE)
    run(rb, [[h0], [wp(pre + [0, 0.02, 0.02], R1, GRIP_CLOSE, vmax=0.5, wmax=0.5, tol_p=0.004, tol_r=0.03),
                    wp(pre, R1, GRIP_CLOSE, vmax=0.3, tol_p=0.0008, tol_r=0.006, hold=6, timeout=60)]], 150, 'pre')
    if rb.done or rb.success:
        return rb.success
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
        if rb.done or rb.success:
            return rb.success
    # insertion: target moves along -y at INSERT_RATE, PI on x/z
    y0 = rb.arm(1)['pos'][1]
    y_end = hole[1] + INSERT_END
    if tip_y is not None and -0.155 < tip_y < -0.125:
        # put the shaft centre (tip + 6 cm) 5 mm past the ring centre
        y_end = hole[1] - 0.005 - (tip_y + 0.06)
    log(f'insert: tip from eef {tip_y}  from y={y0:.4f} to {y_end:.4f} step {rb.step_count}')
    n = max(1, int(np.ceil((y0 - y_end) / INSERT_RATE)))
    ys = list(np.linspace(y0, y_end, n + 1)[1:]) + [y_end] * 12
    res = track(rb, R1, tgt[[0, 2]], ys, h0)
    if res == 'end':
        # not triggered at the planned depth: sweep slowly around it (success window is ~+-1.1 cm)
        log(f'insert end without success at step {rb.step_count}; sweeping')
        ys = list(y_end + 0.001 * np.arange(1, 21)) + list(y_end + 0.02 - 0.001 * np.arange(1, 36))
        res = track(rb, R1, tgt[[0, 2]], ys, h0, 'sweep')
    log(f'insert result {res}: step {rb.step_count} p1={np.round(rb.arm(1)["pos"], 4)} succ={rb.success}')
    return rb.success


def main():
    rb = Robot()
    log(f'start step {rb.step_count}')
    hold_steps(rb, 6, (GRIP_OPEN, GRIP_OPEN))   # let objects settle
    est = perceive_scene(rb)
    if est is None or est.get('ring') is None or est.get('needle') is None:
        # move both arms up and back out of the camera's way, then look again
        a = [rb.arm(0), rb.arm(1)]
        run(rb, [[wp([-0.30, a[i]['pos'][1], 1.20], a[i]['R'], GRIP_OPEN, tol_p=0.01, tol_r=0.1, timeout=60)]
                 for i in range(2)], 80, 'clear')
        est = perceive_scene(rb)
        if est is None or est.get('ring') is None or est.get('needle') is None:
            log('perception failed; giving up')
            return
    tg = update_targets({}, est)
    held = acquire(rb, tg)
    if rb.success or rb.done:
        return
    if not all(held):
        log('could not grasp both objects; continuing anyway', held)
    bring_together(rb, tg)
    if rb.success or rb.done:
        return
    for attempt in range(8):   # every attempt re-measures ring and shaft; keep trying until the horizon
        if align_and_insert(rb, tg, attempt):
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
