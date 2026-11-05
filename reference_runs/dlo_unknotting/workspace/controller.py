"""Frozen feedback controller for the DLO-Lab unknotting task (two Pandas natively attached to the rope).

Strategy (see memory/PROGRESS.md for the experiments behind it):
  1. lift the right gripper P (holds the knot's ring) while the left gripper Q (holds the standing part) feeds toward
     the knot, so the growing ring draws the tail back through the knot while the knot stays on the floor;
  2. keep lifting P while Q pulls back, so the ring draws the last part of the tail out of the knot;
  3. lower P back to the floor while Q retreats along -y (|P - Q| bounded by D_MAX); once the tail is released (~s070),
     P drifts 3 cm along +y while descending, so the standing part lands with ~1.5 cm slack (|P - Q| ~0.64) and lies
     flat instead of buckling next to P, and the released tail falls flat beyond P;
  4. drag both grippers along -y at full speed so the landed tail trails behind P and straightens.
All motions are relative to the initial end-effector positions (the initial scene is randomly translated).
Native action units: translation 0.01 m per unit, per-arm L2 cap 1.0 (0.01 m/step), world frame.
"""
import json
import time

import numpy as np

import robot_api as robot

STEP_M = 0.01      # metres per unit translation command (native)
Z_MIN = 0.03       # lowest commanded EEF height (initial EEF height is ~0.025)
P_Z_FLOOR = 0.0275 # lowest commanded height of P (fingers reach ~0.7 cm below the TCP; floor at z = 0)
Z_MAX = 0.55
D_MAX = 0.65       # max commanded |P - Q|; rope length between the grippers is ~0.66 m (do not over-stretch)
D_EEF_MAX = 0.65   # proprioceptive guard: no arm motion that increases the measured |P - Q| beyond this
P_ROT = (0.0, 0.0, 0.0)  # right-arm rotation per step (native units, 0.1 deg each); kept zero: yaw had no effect and
                         # a tilt/twist brought back the S kink (observed: a[11]=+1 turns the hand clockwise from above)
LOG_PATH = "/tmp/unknot_controller.jsonl"

# Phases: (steps, P velocity, Q velocity) in native units (world frame).
PLAN = [
    (8, (0, 0, 1.0), (0, 0, 0)),
    (30, (0, 0, 0.9), (0, 0.9, 0)),
    (10, (0, 0, 0.6), (0, -1.0, 0)),
    (21, (0, 0, -1.0), (0, -1.0, 0)),
    (12, (0, 0.25, -0.9682), (0, -1.0, 0)),       # after the release: +3 cm y for 0.38 cm less descent
    (8, (0, 0, -1.0), (0, -1.0, 0)),
    (4, (0, -0.9955, -0.0953), (0, -1.0, 0)),     # drag, finishing the remaining 0.38 cm of descent
    (7, (0, -1.0, 0), (0, -1.0, 0)),
]


def _cap(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1.0 else v


class Commander:
    """Tracks the accumulated IK targets and keeps them inside safe bounds relative to the initial poses."""

    def __init__(self, obs, d_max=D_MAX, d_eef_max=D_EEF_MAX, p_z_floor=P_Z_FLOOR):
        self.d_max = d_max
        self.p_z_floor = p_z_floor
        self.d_eef_max = d_eef_max
        self.guarded = 0
        self.P0 = np.array(obs["robot"]["right"]["eef_position_world"], np.float64)
        self.Q0 = np.array(obs["robot"]["left"]["eef_position_world"], np.float64)
        self.tP = self.P0.copy()
        self.tQ = self.Q0.copy()

    def _bounded(self, t, v, lo, hi):
        new = np.clip(t + _cap(v) * STEP_M, lo, hi)
        return (new - t) / STEP_M, new

    def action(self, vP, vQ, P_eef=None, Q_eef=None):
        loP = np.array([-np.inf, -np.inf, min(Z_MIN, self.P0[2], self.p_z_floor)]); hiP = np.array([np.inf, np.inf, Z_MAX])
        loQ = np.array([-np.inf, -np.inf, min(Z_MIN, self.Q0[2])]); hiQ = np.array([np.inf, np.inf, Z_MAX])
        aP, tP = self._bounded(self.tP, vP, loP, hiP)
        aQ, tQ = self._bounded(self.tQ, vQ, loQ, hiQ)
        if np.linalg.norm(tP - tQ) > self.d_max and np.linalg.norm(tP - tQ) > np.linalg.norm(self.tP - self.tQ):
            aQ, tQ = np.zeros(3), self.tQ.copy()
            if np.linalg.norm(tP - tQ) > self.d_max:
                aP, tP = np.zeros(3), self.tP.copy()
        if P_eef is not None and Q_eef is not None:
            D = np.asarray(P_eef) - np.asarray(Q_eef)
            if np.linalg.norm(D) >= self.d_eef_max:
                u = D / np.linalg.norm(D)
                self.guarded += 1
                if u @ aQ < 0:
                    aQ, tQ = np.zeros(3), self.tQ.copy()
                if u @ aP > 0:
                    aP, tP = np.zeros(3), self.tP.copy()
        self.tP, self.tQ = tP, tQ
        a = np.zeros(12)
        a[0:3] = aQ
        a[3:6] = aP
        return np.clip(a, -1.0, 1.0)


def _eefs(obs):
    return (np.array(obs["robot"]["right"]["eef_position_world"]), np.array(obs["robot"]["left"]["eef_position_world"]))


def main(plan=None, on_step=None, log_path=LOG_PATH, d_max=D_MAX, d_eef_max=D_EEF_MAX, p_z_floor=P_Z_FLOOR, p_rot=P_ROT):
    plan = PLAN if plan is None else plan
    obs = robot.observe()
    cmd = Commander(obs, d_max, d_eef_max, p_z_floor)
    P, Q = _eefs(obs)
    try:
        log = open(log_path, "a")
    except OSError:
        log = None
    res = {"observation": obs, "terminal": bool(obs.get("terminal")), "final_native_metric": obs.get("final_native_metric")}
    k = 0
    schedule = [(vP, vQ) for n, vP, vQ in plan for _ in range(int(n))]
    while not res.get("terminal"):
        vP, vQ = schedule[k] if k < len(schedule) else ((0, 0, 0), (0, 0, 0))
        a = cmd.action(vP, vQ, P, Q)
        a[9:12] = np.clip(p_rot, -1.0, 1.0)
        res = robot.step(a.tolist())
        k += 1
        o = res.get("observation", {})
        if "robot" in o:
            P, Q = _eefs(o)
        if log is not None:
            log.write(json.dumps({"t": time.time(), "k": k, "step": o.get("native_step"), "a": np.round(a, 3).tolist(),
                                  "P": np.round(P, 4).tolist(), "Q": np.round(Q, 4).tolist(),
                                  "guarded": cmd.guarded, "terminal": res.get("terminal"),
                                  "metric": res.get("final_native_metric")}) + "\n")
            log.flush()
        if on_step is not None:
            on_step(k, res)
        if k > 400:  # the native horizon is 100 steps; never loop forever
            break
    if log is not None:
        log.close()
    return res


if __name__ == "__main__":
    main()
