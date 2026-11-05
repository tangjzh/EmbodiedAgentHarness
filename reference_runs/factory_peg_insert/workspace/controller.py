"""Factory peg insertion controller: RGB-D socket/peg localisation + compliant descent.

Runs one complete native episode (to the task horizon) using only robot_api sensors.
Diagnostics go to /tmp/peg_insert_log.json.
"""
import json
import time

import numpy as np
import robot_api as robot

import perception

LOG = "/tmp/peg_insert_log.json"
POS_SCALE = 0.02          # metres per unit action (native)
HOVER = 0.008             # peg tip clearance above socket top while aligning (m)
XY_TOL = 0.0006


def _obs():
    o = robot.observe()
    r = o["robot"]
    return o, np.asarray(r["eef_position_world"], float)


def perceive():
    o, eef = _obs()
    est = perception.estimate(robot.cameras(), o["cameras"], eef[:2])
    est["eef"] = eef.tolist()
    return o, eef, est


class Runner:
    def __init__(self, horizon):
        self.horizon = horizon
        self.log = {"steps": [], "events": []}
        self.done = False
        self.step_i = 0

    def step(self, a):
        if self.done:
            return None
        a = np.clip(np.asarray(a, float), -1, 1)
        fb = robot.step(a.tolist())
        self.step_i += 1
        o = fb if isinstance(fb, dict) and "robot" in fb else robot.observe()
        eef = np.asarray(o["robot"]["eef_position_world"], float)
        self.log["steps"].append({"i": self.step_i, "a": np.round(a, 3).tolist(), "eef": np.round(eef, 5).tolist(),
                                  "succ": bool(o.get("native_success", False))})
        if o.get("terminal") or self.step_i >= self.horizon:
            self.done = True
            self.log["final"] = {k: o.get(k) for k in ("native_step", "native_success", "final_native_metric", "terminal")}
        return eef

    def event(self, **kw):
        kw["step"] = self.step_i
        self.log["events"].append(kw)

    def dump(self):
        with open(LOG, "w") as f:
            json.dump(self.log, f, default=lambda x: np.asarray(x).tolist())


def servo_action(eef, target, vel, gain=0.45, damp=1.2):
    """PD on EEF position; the native EMA (0.2) lags actions, so keep gains low and damp with velocity."""
    a = np.zeros(6)
    a[:3] = (gain * (target - eef) - damp * vel) / POS_SCALE
    return a


def main():
    task = robot.task()
    horizon = int(task.get("horizon", 150))
    R = Runner(horizon)
    t0 = time.time()
    try:
        run(R)
    finally:
        R.log["wall_s"] = time.time() - t0
        R.dump()
    # run to native termination
    while not R.done:
        R.step(np.zeros(6))
    R.dump()


def localise(R, est_prev=None):
    o, eef, est = perceive()
    ok = "hole_xy" in est and "peg_tip_z" in est
    R.event(kind="perceive", est=est)
    return eef, est, ok


def run(R):
    eef, est, ok = localise(R)
    if not ok:
        # fall back on socket face centre if the bore is not segmented
        est.setdefault("hole_xy", est.get("socket_xy", eef[:2].tolist()))
    hole = np.asarray(est["hole_xy"])
    top_z = est.get("socket_top_z", eef[2] - 0.06)
    peg_off = np.asarray(est.get("peg_axis_xy", eef[:2])) - eef[:2]   # peg axis minus eef, XY
    if np.linalg.norm(peg_off) > 0.004:
        peg_off = np.zeros(2)
    tip_off = est.get("peg_tip_z", eef[2] - 0.032) - eef[2]           # tip z minus eef z (negative)
    R.event(kind="plan", hole=hole, top_z=top_z, peg_off=peg_off, tip_off=tip_off)

    prev = eef.copy()
    search = [np.zeros(2)] + [0.0008 * np.array([np.cos(t), np.sin(t)]) for t in np.arange(8) * np.pi / 4]
    attempt = 0
    while not R.done:
        xy = hole - peg_off + search[attempt % len(search)]
        # Phase 1: hover above the bore until XY settles
        tgt = np.array([xy[0], xy[1], top_z + HOVER - tip_off])
        settled = 0
        for _ in range(45):
            vel = eef - prev; prev = eef
            eef = R.step(servo_action(eef, tgt, vel))
            if eef is None:
                return
            settled = settled + 1 if np.linalg.norm((eef - tgt)[:2]) < XY_TOL and np.linalg.norm(eef - prev) < 0.0004 else 0
            if settled >= 3:
                break
        R.event(kind="aligned", attempt=attempt, eef=eef, err=eef - tgt)

        # Phase 2: descend with XY servo, commanding below the bore bottom so the impedance presses in
        stall = 0
        best_z = eef[2]
        while not R.done:
            vel = eef - prev; prev = eef
            tgt = np.array([xy[0], xy[1], max(top_z - 0.03 - tip_off, eef[2] - 0.006)])
            a = servo_action(eef, tgt, vel)
            a[2] = -0.3
            eef = R.step(a)
            if eef is None:
                return
            tip = eef[2] + tip_off
            if eef[2] < best_z - 0.0003:
                best_z = eef[2]; stall = 0
            else:
                stall += 1
            # jammed on the rim (tip not yet past the chamfer): lift and try a nearby offset
            if stall > 10 and tip > top_z - 0.004:
                R.event(kind="jam", tip=tip, eef=eef)
                attempt += 1
                break
            if stall > 10:
                stall = -10 ** 6  # inserted; keep pressing to the horizon
        else:
            return
