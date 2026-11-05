"""Factory GearMesh controller: perceive peg with RGB-D, align held gear, insert.

main() runs one complete episode to native termination using only sensors,
proprioception and the read-only task spec. Diagnostics go to /tmp.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robot_api as robot  # noqa: E402
from gearmesh.perception import world_points, plate_height, find_peg, held_gear  # noqa: E402

LOG = '/tmp/gearmesh_log.jsonl'
POS_SCALE = 0.02          # metres per unit action (incremental, world frame)
ROT_SCALE = 0.097         # rad per unit action
GEAR_BOTTOM_BELOW_EEF = 0.012   # measured: blue gear lowest points ~12 mm below EEF
PEG_CLEAR = 0.006
TEST_YAW_OFFSET = 0.0     # development only: deliberately mis-phase teeth to exercise the sweep
YAW_SWEEP = 0.10          # rad either side of the stall yaw (> half a tooth pitch)
VEL_PER_ACTION = 0.0044   # steady EEF speed (m/step) per unit action, from logs         # hover clearance of gear bottom above peg top


def _js(o):
    return o.tolist() if hasattr(o, 'tolist') else str(o)


def log(rec):
    try:
        with open(LOG, 'a') as f:
            f.write(json.dumps(rec, default=_js) + '\n')
    except Exception:
        pass


def wrap(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def yaw_of(q):
    w, x, y, z = q
    # hand is roll=pi; yaw of the hand x-axis projected in world
    R00 = 1 - 2 * (y * y + z * z)
    R10 = 2 * (x * y + w * z)
    return float(np.arctan2(R10, R00))


class Runner:
    def __init__(self):
        self.obs = robot.observe()
        try:
            self.horizon = int(robot.task().get('horizon', 300))
        except Exception:
            self.horizon = 300
        self.done = bool(self.obs.get('terminal'))
        self.t = int(self.obs.get('native_step', 0))
        self.success_any = False
        self.vel = np.zeros(3)
        self.yaw_vel = 0.0

    @property
    def yaw(self):
        return yaw_of(self.obs['robot']['eef_quaternion_wxyz'])

    @property
    def eef(self):
        return np.asarray(self.obs['robot']['eef_position_world'], float)

    def step(self, a, phase=''):
        a = np.clip(np.asarray(a, float), -1, 1)
        r = robot.step(a.tolist(), repeat=1)
        prev, prev_yaw = self.eef, self.yaw
        self.obs = r['observation']
        self.vel = self.eef - prev
        self.yaw_vel = wrap(self.yaw - prev_yaw)
        self.done = bool(r.get('terminal'))
        self.t = int(self.obs.get('native_step', self.t + 1))
        ns = bool(r.get('native_success'))
        self.success_any |= ns
        log(dict(t=self.t, phase=phase, a=a.round(3).tolist(), eef=self.eef.round(5).tolist(),
                 yaw=round(yaw_of(self.obs['robot']['eef_quaternion_wxyz']), 4), succ=ns))
        return r


def perceive(obs, tag='init'):
    cams = robot.cameras()
    try:
        np.savez_compressed(f'/tmp/gearmesh_{tag}.npz', **cams)
        with open(f'/tmp/gearmesh_{tag}_obs.json', 'w') as f:
            json.dump(obs, f)
    except Exception:
        pass
    P, C = world_points(obs, cams)
    eef = np.asarray(obs['robot']['eef_position_world'], float)
    pz = plate_height(P, eef)
    peg = find_peg(P, eef[:2], pz)
    gear = held_gear(P, C, eef)
    return dict(plate_z=pz, peg=peg, gear=gear)


KP, KD = 0.5, 1.5   # tuned offline on an ARX model fitted to logged motion


def pos_action(err, vel=0.0, kp=KP, kd=KD, lim=1.0):
    """PD on position; the executor integrates actions (action ~ velocity) with ~6 step lag."""
    return np.clip(kp * np.asarray(err) / POS_SCALE - kd * np.asarray(vel) / POS_SCALE, -lim, lim)


def move_to(R, xy, z, tol, t_max, phase='align', yaw=None):
    """PD move until xy within tol, z within 2 mm and nearly still (3 steps)."""
    settle = 0
    while not R.done and R.t < t_max:
        e = R.eef
        err = np.r_[xy - e[:2], z - e[2]]
        a_yaw = 0.0
        if yaw is not None:
            a_yaw = np.clip(KP * wrap(yaw - R.yaw) / ROT_SCALE - KD * R.yaw_vel / ROT_SCALE, -0.5, 0.5)
        R.step(np.r_[pos_action(err, R.vel), 0, 0, a_yaw], phase)
        ok = np.hypot(*err[:2]) < tol and np.hypot(*R.vel[:2]) < 0.0002 and abs(err[2]) < 0.002
        settle = settle + 1 if ok else 0
        if settle >= 3:
            return True
    return False


def main():
    try:
        open(LOG, 'w').close()
    except Exception:
        pass
    R = Runner()
    try:
        run(R)
    except Exception as exc:  # never leave the episode before native termination
        log(dict(error=repr(exc)))
    finally:
        while not R.done:
            try:
                R.step(np.zeros(6), 'hold')
            except Exception as exc:
                log(dict(hold_error=repr(exc)))
                break
    log(dict(end=True, t=R.t, eef=R.eef.tolist(), success_any=R.success_any,
             native_success=R.obs.get('native_success'), final_metric=R.obs.get('final_native_metric')))
    print(json.dumps(dict(t=R.t, eef=R.eef.round(4).tolist(), success_any=R.success_any,
                          native_success=R.obs.get('native_success'),
                          final_metric=R.obs.get('final_native_metric'))))


def safe_perceive(obs, tag='init'):
    try:
        return perceive(obs, tag)
    except Exception as exc:
        log(dict(perception_error=repr(exc)))
        eef = np.asarray(obs['robot']['eef_position_world'], float)
        # typical native pre-grasp: gear face ~33 mm above the 20 mm peg on the plate
        return dict(plate_z=eef[2] - 0.065, peg=dict(ok=False), gear={})


def run(R):
    per = safe_perceive(R.obs)
    if not per['peg'].get('ok'):
        # peg probably occluded by the held gear: rise a little and look again
        z_up = R.eef[2] + 0.025
        for _ in range(20):
            R.step(np.r_[pos_action(np.r_[0, 0, z_up - R.eef[2]], R.vel), 0, 0, 0], 'reperceive')
        per = safe_perceive(R.obs, 'retry')
    peg = per['peg']
    log(dict(perception={k: (v if k != 'peg' else {kk: vv for kk, vv in v.items() if kk != 'cands'})
                         for k, v in per.items()}))
    if peg.get('ok'):
        peg_xy = np.asarray(peg['xy'], float)
        peg_top = float(peg['top_z'])
    else:  # fall back: insert straight down from where the native pre-grasp put us
        peg_xy = R.eef[:2].copy()
        peg_top = per['plate_z'] + 0.02
    g = per['gear']
    off = np.asarray(g['offset'], float) if g.get('offset') is not None else np.zeros(2)
    if np.hypot(*off) > 0.006:
        off = np.zeros(2)
    bottom = g['bottom'] if g.get('bottom') and 0.006 < g['bottom'] < 0.02 else GEAR_BOTTOM_BELOW_EEF
    goal_xy = peg_xy - off          # EEF xy that puts the gear axis on the peg axis
    tip_z = peg_top + bottom        # EEF z where the gear's lower face reaches the peg top
    hover_z = tip_z + PEG_CLEAR
    final_z = per['plate_z'] + bottom
    log(dict(plan=dict(goal_xy=goal_xy, tip_z=tip_z, final_z=final_z, off=off, bottom=bottom)))

    # Phase 1: align xy above the peg, staying clear of the neighbouring gears,
    # then lower to just above the peg tip and settle precisely.
    stage1_z = np.clip(R.eef[2], hover_z + 0.008, hover_z + 0.03)
    move_to(R, goal_xy, stage1_z, tol=0.001, t_max=R.t + 90, yaw=R.yaw + TEST_YAW_OFFSET)
    move_to(R, goal_xy, tip_z + 0.0025, tol=0.0004, t_max=R.t + 45)

    # Phase 2: descend while holding xy. Stalled on the peg tip -> lift slightly
    # (frees friction), step to the next golden-angle spiral point, retry.
    # Stalled lower (teeth not in phase with the fixed gears) -> sweep yaw slowly
    # back and forth under a light push until the teeth drop into mesh.
    z_cmd = R.eef[2]
    best_z, stall, tries = R.eef[2], 0, 0
    sweep_dir, yaw0 = 0, None
    tgt = goal_xy.copy()
    while not R.done:
        e = R.eef
        yaw = R.yaw
        rate = 0.0015 if z_cmd > tip_z + 0.004 else 0.0006
        z_cmd = max(final_z - 0.004, z_cmd - rate)
        v_ref = -rate if z_cmd > final_z - 0.004 else 0.0
        if e[2] < best_z - 0.0003:
            best_z, stall = e[2], 0
        elif z_cmd < e[2] - 0.002:      # only count once we are really pushing down
            stall += 1
        a = np.zeros(6)
        a[:2] = pos_action(tgt - e[:2], R.vel[:2])
        a[2] = np.clip(v_ref / VEL_PER_ACTION + pos_action(z_cmd - e[2], R.vel[2] - v_ref), -0.4, 0.4)
        stalled = e[2] > final_z + 0.0015 and stall > 5
        if stalled and abs(e[2] - tip_z) < 0.003 and R.horizon - R.t > 25:
            tries += 1
            r = min(0.0025, 0.0006 * np.sqrt(tries))
            tgt = goal_xy + r * np.array([np.cos(2.4 * tries), np.sin(2.4 * tries)])
            log(dict(retry=tries, tgt=tgt))
            move_to(R, tgt, tip_z + 0.0025, tol=0.0004, t_max=R.t + 25, phase='retry')
            z_cmd, best_z, stall = R.eef[2], R.eef[2], 0
            continue
        if stalled:
            if sweep_dir == 0:
                sweep_dir, yaw0 = 1, yaw
            d = wrap(yaw - yaw0)
            if d * sweep_dir > YAW_SWEEP:
                sweep_dir = -sweep_dir
            a[5] = 0.5 * sweep_dir
            a[2] = -0.15
        R.step(a, 'insert')


if __name__ == '__main__':
    main()
