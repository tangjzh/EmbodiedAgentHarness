"""hammer_nail controller: pick up the hammer and drive the nail.

State machine (each step re-perceives, acts, verifies; see TRANSITIONS.md):
  SCENE   -> table, board, nail, coarse hammer from the cameras          -> SURVEY (fail: VIEW pose, retry)
  SURVEY  -> wrist camera looks at the hammer; fit handle axis/head/incline, twice until the
             hammer has settled                                           -> GRASP (fail: SURVEY wider, then
             front-only fit, then SCENE)
  GRASP   -> palm-down power grasp across the handle; verify finger blocking -> LIFT (fail: RELEASE -> SCENE)
  LIFT    -> lift a little, verify grip, lift to carry height, verify hammer left the table
                                                                          -> CARRY (fail: RELEASE -> SCENE)
  CARRY   -> carry above the nail, level palm at the strike yaw, measure face, align -> STRIKE
  STRIKE  -> step target far below, 1 row/call; task_complete ends episode   -> re-measure nail+face, realign,
             STRIKE again (hammer lost: RELEASE -> SCENE)
"""
import os
import sys

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")   # host BLAS threading is pathologically slow (memory/PROGRESS.md)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)       # the runner imports controller.py without our directory on the path

import json
import math
import traceback

import numpy as np
from scipy.spatial.transform import Rotation as R

import perception as pc
from motion import (HAND_OPEN, continuous_rotvec, grasp_rotation, interp_rows, move_steps,
                    palm_down, row, yaw_of)
from robot import EpisodeOver, Log, Robot

VERSION = "v2"

# ---- tunables (E1 manual, E2 controller success; others untested) ----
BASE_XY = np.array([-0.8, 0.0])        # Panda base (robot-interface skill)
SURVEY_D = 0.42                        # wrist-camera distance to hammer centre (E2 1/1)
SURVEY_MIN_REACH = 0.30                # survey flange at least this far (horizontally) from the base (untested)
CAM_OFF_F = np.array([-0.14, -0.03, 0.04])   # wrist camera in flange frame (E1 calibration)
CAM_DIR_F = np.array([0.865, 0.0, 0.501])    # wrist optical axis in flange frame (E1 calibration)
GRASP_OFFSET_F = np.array([0.085, 0.0, 0.055])  # handle centre in flange frame (E1, E2 2/2)
A_GRASP = 0.095                        # grasp point from head-side end of handle (E1, E2)
PRE_GRASP_UP = 0.08                    # pre-grasp along palm normal (E1, E2)
CLOSE1 = [0, 0.5, 1.2, 1.2] * 3 + [1.2, 0.3, 0.6, 0.6]     # E1, E2
CLOSE2 = [0, 0.9, 1.5, 1.5] * 3 + [1.3, 0.6, 1.0, 1.0]     # E1, E2 squeeze
FACE_PRIOR_F = np.array([0.060, -0.142, 0.162])  # face (lowest slab) in flange frame, index side (E1, E2)
STRIKE_H = 0.07                        # face height above nail top before the strike (E1, E2)
STRIKE_DZ = 0.26                       # target step below the pre-strike pose (E1, E2 2/2)
STRIKE_MAX_ROWS = 30
ALIGN_TOL = 0.005                      # face-nail xy error accepted (E2 converged to 1 mm)
MAX_YAW_DEV = math.radians(100)        # index-side grasp allowed this far from outward yaw (untested)
# strike yaw window relative to the nail's outward direction: keeps the head (flange -y for index side)
# outward of the flange and the fingers from pointing back at the base (untested beyond E1/E2: -61 deg)
STRIKE_YAW_WIN = {"index": (math.radians(-100), math.radians(-7)),
                  "pinky": (math.radians(7), math.radians(100))}
MAX_STRIKES_PER_GRASP = 8
# speeds (m/s, rad/s): free arm and carrying the hammer (E2 v1 used 0.25/1.0 everywhere)
V_FREE, W_FREE = 0.40, 1.5
V_CARRY, W_CARRY = 0.45, 1.0
SETTLE_HOLD = 10                       # rows between the two survey fits used to confirm the hammer settled
VIEW_POSE = np.array([-0.443, 0.0, 1.517])   # start pose: arm out of the front camera's view of the table


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def fits_agree(a, b):
    """Two handle fits describe the same, resting hammer."""
    dc = float(np.linalg.norm(np.asarray(a["c"]) - np.asarray(b["c"])))
    cosd = abs(float(np.dot(a["d"], b["d"])))
    return (dc < 0.015 and cosd > math.cos(math.radians(6)) and abs(a["incline_deg"] - b["incline_deg"]) < 3.0
            and abs(a["z0"] - b["z0"]) < 0.012)


class Controller:
    def __init__(self, ctx, stop_after=None):
        self.log = Log()
        self.rb = Robot(ctx, self.log)
        self.stop_after = stop_after       # development only: return after this state
        self.scene = None
        self.fit = None
        self.grasp = None
        self.face_f = None
        self.cmd = None
        self.strike_yaw = None
        self.stats = {"version": VERSION, "strikes": 0, "grasp_attempts": 0, "states": [], "phase_steps": {}}

    # ------------------------------------------------------------ helpers
    def L(self, *a, **k):
        self.log(f"[{VERSION} step {self.rb.step_id} rem {self.rb.remaining}]", *a, **k)

    def move(self, p1, r1, hand=None, n=None, hold=10, v_max=V_FREE, w_max=W_FREE):
        """Smooth move from the last commanded target to (p1, r1, hand), then hold."""
        cur = self.rb.hold_row()
        p0 = np.asarray(cur[0:3])
        r0 = R.from_rotvec(cur[3:6])
        h0 = cur[6:22]
        h1 = h0 if hand is None else hand
        if n is None:
            n = move_steps(p0, r0, p1, r1, v_max=v_max, w_max=w_max)
        rows = interp_rows(p0, r0, h0, p1, r1, h1, n, near=np.asarray(cur[3:6]))
        rows += [rows[-1]] * int(hold)
        return self.rb.step(rows)

    def hand_to(self, hand, n=20, hold=10):
        cur = self.rb.hold_row()
        h0 = np.asarray(cur[6:22])
        h1 = np.asarray(hand, float)
        rows = []
        for i in range(1, n + 1):
            s = i / n
            rows.append(cur[0:6] + list((1 - s) * h0 + s * h1))
        rows += [rows[-1]] * hold
        return self.rb.step(rows)

    def blocking(self, state, target=CLOSE2):
        h = np.asarray(state[7:23])
        t = np.asarray(target)
        deficit = {}
        for name, k in (("ff", 0), ("mf", 4), ("rf", 8), ("th", 12)):
            deficit[name] = round(float(np.sum(np.clip(t[k + 1:k + 4] - h[k + 1:k + 4], 0, None))), 3)
        n = sum(1 for name in ("ff", "mf", "th") if deficit[name] > 0.15)
        return {"n_blocked": n, "deficit": deficit}

    def holding_hammer(self):
        h = (self.rb.target or [0.0] * 22)[6:22]
        return float(np.max(np.abs(np.asarray(h[0:12])))) > 0.3

    # ------------------------------------------------------------ SCENE
    def perceive_scene(self, obs):
        tz = pc.table_height(obs)
        if tz is None:
            tz = self.scene["table_z"] if self.scene else 0.945
            self.L("WARN table not found", table_z=tz)
        board = pc.board_info(obs, tz)
        if board is None:
            board = (self.scene or {}).get("board") or {"top": tz + 0.110, "x": [-0.34, 0.05], "y": [-0.20, 0.20], "n": 0}
            self.L("WARN board not found", board=board)
        fp = np.asarray(obs.state[0:3])
        nail = pc.nail_info(obs, board, exclude_above=fp[2] - 0.12)
        if nail is None and self.scene and self.scene.get("nail"):
            nail = self.scene["nail"]
        z_cap = fp[2] - 0.10
        wood, grey = pc.hammer_points(obs, tz, board, cams=("front",), z_cap=z_cap)
        coarse = None
        if len(wood) + len(grey) >= 40:
            pts = pc.largest_cluster_xy(np.concatenate([wood, grey]))
            coarse = {"c": pts.mean(0).tolist(), "n_wood": int(len(wood)), "n_grey": int(len(grey))}
        cfit = pc.fit_handle(wood, grey)
        self.scene = {"table_z": tz, "board": board, "nail": nail, "coarse": coarse, "coarse_fit": cfit}
        self.L("SCENE", table_z=tz, board=board, nail=nail, coarse=coarse,
               coarse_fit=None if cfit is None else {k: cfit[k] for k in ("c", "d", "incline_deg", "length", "n_top", "head")})
        return nail is not None and coarse is not None

    def go_view_pose(self):
        """Move the (empty) arm to the start pose so the front camera sees the table."""
        if self.holding_hammer():
            self.release_and_clear()
        self.move(VIEW_POSE, palm_down(0.0), hand=HAND_OPEN, hold=10)

    # ------------------------------------------------------------ yaw choice
    def choose_rotation(self, d, incline, c):
        """Index-side grasp (head toward flange -y) unless its yaw is far from the outward yaw."""
        rot_a = grasp_rotation(d, incline)
        rot_b = grasp_rotation(-np.asarray(d), -incline)
        out = math.atan2(c[1] - BASE_XY[1], c[0] - BASE_XY[0])
        dev_a = abs(wrap(yaw_of(rot_a) - out))
        if dev_a <= MAX_YAW_DEV:
            return rot_a, "index", dev_a
        return rot_b, "pinky", abs(wrap(yaw_of(rot_b) - out))

    def nail_out(self):
        nail = self.scene["nail"]
        return math.atan2(nail["y"] - BASE_XY[1], nail["x"] - BASE_XY[0])

    def choose_strike_yaw(self):
        """Grasp yaw clamped into the side's window around the nail's outward direction (never through
        the fingers-back direction)."""
        out = self.nail_out()
        lo, hi = STRIKE_YAW_WIN[self.grasp["side"]]
        rel = wrap(self.grasp["yaw"] - out)
        return wrap(out + min(max(rel, lo), hi))

    # ------------------------------------------------------------ SURVEY
    def survey_fit(self, obs):
        sc = self.scene
        fp = np.asarray(obs.state[0:3])
        wood, grey = pc.hammer_points(obs, sc["table_z"], sc["board"], cams=("front", "wrist"), z_cap=fp[2] - 0.10)
        fit = pc.fit_handle(wood, grey)
        ok = (fit is not None and fit["n_top"] >= 80 and 0.10 <= fit["length"] <= 0.28
              and 0.015 <= fit["width"] <= 0.08 and abs(fit["incline_deg"]) < 35)
        return fit, ok

    def survey(self, attempt):
        sc = self.scene
        tz = sc["table_z"]
        src = self.fit if (self.fit is not None and attempt > 0) else sc.get("coarse_fit")
        c = np.asarray(src["c"]) if src else np.asarray(sc["coarse"]["c"][:2])
        if src:
            rot, side, dev = self.choose_rotation(src["d"], 0.0, c)
            yaw = yaw_of(rot)
        else:
            yaw = math.atan2(c[1] - BASE_XY[1], c[0] - BASE_XY[0])
        T = np.array([c[0], c[1], tz + 0.04])
        r = palm_down(yaw)
        D = SURVEY_D + 0.06 * attempt
        while True:   # keep the survey flange away from the base (reach)
            C = T - D * r.apply(CAM_DIR_F)
            F = C - r.apply(CAM_OFF_F)
            if np.linalg.norm(F[:2] - BASE_XY) >= SURVEY_MIN_REACH or D <= 0.30:
                break
            D -= 0.03
        self.L("SURVEY move", attempt=attempt, target=F, yaw_deg=math.degrees(yaw), D=D)
        obs = self.move(F, r, hand=HAND_OPEN, hold=8)
        self.L("SURVEY reached", flange=obs.state[0:3], err=np.asarray(obs.state[0:3]) - F)
        prev = None
        for k in range(5):
            if k > 0:
                obs = self.rb.hold(SETTLE_HOLD)
            fit, ok = self.survey_fit(obs)
            agree = bool(ok and prev is not None and fits_agree(prev, fit))
            self.L("SURVEY fit", k=k, ok=ok, agree=agree, fit=fit)
            if agree:
                self.fit = fit
                return True
            if ok:
                prev = fit
            elif k >= 1 and prev is None:
                break
        if prev is not None:
            self.fit = prev            # never confirmed settled; better than nothing
            return True
        return False

    # ------------------------------------------------------------ GRASP
    def do_grasp(self, attempt):
        fit = self.fit
        a_g = fit["a_min"] + A_GRASP
        a_g = min(max(a_g, fit["a_min"] + 0.07), fit["a_max"] - 0.04)
        g = pc.handle_point(fit, a_g)
        g[2] -= 0.004 * min(attempt, 2)          # escalation: grasp slightly deeper on retries
        incline = math.radians(fit["incline_deg"])
        rot, side, dev = self.choose_rotation(fit["d"], incline, np.asarray(fit["c"]))
        F = g - rot.apply(GRASP_OFFSET_F)
        F_pre = F - PRE_GRASP_UP * rot.apply([0, 0, 1])
        self.grasp = {"g": g, "F": F, "rot": rot, "side": side, "yaw": yaw_of(rot), "a_g": a_g}
        self.L("GRASP plan", g=g, F=F, F_pre=F_pre, side=side, yaw_deg=math.degrees(yaw_of(rot)),
               yaw_dev_deg=math.degrees(dev), incline_deg=fit["incline_deg"])
        self.move(F_pre, rot, hand=HAND_OPEN, hold=8, v_max=0.35)
        obs = self.move(F, rot, hand=HAND_OPEN, n=30, hold=10)
        self.L("GRASP at handle", flange=obs.state[0:3], err=np.asarray(obs.state[0:3]) - F)
        self.hand_to(CLOSE1, n=15, hold=5)
        obs = self.hand_to(CLOSE2, n=15, hold=8)
        blk = self.blocking(obs.state)
        ok = blk["n_blocked"] >= 2
        self.L("GRASP closed", ok=ok, blocking=blk, hand=np.round(obs.state[7:23], 2))
        return ok

    # ------------------------------------------------------------ LIFT
    def carry_pose(self):
        """(pre-strike flange target 4 cm above the goal, strike rotation, safe travel height)."""
        nail = self.scene["nail"]
        nail_p = np.array([nail["x"], nail["y"], nail["z"]])
        if self.strike_yaw is None:
            self.strike_yaw = self.choose_strike_yaw()
        r = palm_down(self.strike_yaw)
        s = self.face_prior() if self.face_f is None else self.face_f
        F = nail_p + np.array([0, 0, STRIKE_H + 0.04]) - r.apply(s)
        return F, r, F[2]          # travel at the pre-strike height: face 11 cm above the nail top

    def do_lift(self):
        gr = self.grasp
        F = np.asarray(gr["F"])
        obs = self.move(F + np.array([0, 0, 0.04]), gr["rot"], n=15, hold=5)
        blk = self.blocking(obs.state)
        if blk["n_blocked"] < 2:
            self.L("LIFT grip lost at 4 cm", blocking=blk)
            return False
        _, _, high = self.carry_pose()
        high = max(high, F[2] + 0.15)
        obs = self.move(np.array([F[0], F[1], high]), gr["rot"], hold=6, v_max=V_CARRY, w_max=W_CARRY)
        blk = self.blocking(obs.state)
        sc = self.scene
        wood, grey = pc.hammer_points(obs, sc["table_z"], sc["board"], cams=("front",),
                                      z_cap=sc["table_z"] + 0.12)
        c = np.asarray(self.fit["c"])
        left = 0
        for pts in (wood, grey):
            if len(pts):
                left += int(np.sum(np.linalg.norm(pts[:, :2] - c, axis=1) < 0.15))
        ok = blk["n_blocked"] >= 2 and left < 150
        self.L("LIFT check", ok=ok, blocking=blk, table_pts_left=left, flange=obs.state[0:3], high=high)
        return ok

    # ------------------------------------------------------------ face measurement
    def measure_face(self, obs):
        """Face (lowest 6 mm of the head) in world and flange frames, from both cameras."""
        sc = self.scene
        nail_z = sc["nail"]["z"] if sc.get("nail") else sc["board"]["top"] + 0.06
        fp, Rf = pc.flange_pose(obs.state)
        pts_all = []
        for cam in ("front", "wrist"):
            if not obs.has(cam):
                continue
            P = obs.points(cam)
            r, g, b = obs.colors(cam)
            m = pc.grey_mask(r, g, b) & np.isfinite(P[..., 2])
            m &= np.linalg.norm(P - fp, axis=-1) < 0.30
            m &= P[..., 2] > nail_z + 0.012
            pts = P[m]
            if len(pts):
                Lf = (pts - fp) @ Rf
                pts_all.append(pts[(Lf[:, 2] > 0.03) & (np.abs(Lf[:, 1]) > 0.04)])
        if not pts_all:
            return None
        W = np.concatenate(pts_all)
        if len(W) < 100:
            return None
        zmin = np.percentile(W[:, 2], 1)
        slab = W[W[:, 2] < zmin + 0.006]
        if len(slab) < 10:
            return None
        lo = np.percentile(slab[:, :2], 5, axis=0)
        hi = np.percentile(slab[:, :2], 95, axis=0)
        xy = (lo + hi) / 2
        face_w = np.array([xy[0], xy[1], zmin])
        face_f = (face_w - fp) @ Rf
        plausible = 0.10 < face_f[2] < 0.26
        return {"w": face_w, "f": face_f, "n": int(len(W)), "n_slab": int(len(slab)), "ok": bool(plausible)}

    # ------------------------------------------------------------ CARRY / ALIGN
    def face_prior(self):
        s = FACE_PRIOR_F.copy()
        if self.grasp and self.grasp["side"] == "pinky":
            s[1] = -s[1]
        return s

    def strike_rot(self):
        return palm_down(self.strike_yaw if self.strike_yaw is not None else self.grasp["yaw"])

    def carry(self):
        F, r, high = self.carry_pose()
        cur = np.asarray(self.rb.hold_row()[0:3])
        high = max(cur[2], high)
        self.L("CARRY", via=[cur[0], cur[1], high], to=F, strike_yaw_deg=math.degrees(self.strike_yaw),
               grasp_yaw_deg=math.degrees(self.grasp["yaw"]))
        if high > cur[2] + 0.01:
            cur_r = R.from_rotvec(self.rb.hold_row()[3:6])
            self.move(np.array([cur[0], cur[1], high]), cur_r, hold=0, v_max=V_CARRY, w_max=W_CARRY)
        cur_r = R.from_rotvec(self.rb.hold_row()[3:6])
        if (r * cur_r.inv()).magnitude() > math.radians(120):
            # split big yaw changes so the slerp keeps to the planned (never fingers-back) direction
            out = self.nail_out()
            mid_yaw = out + (wrap(yaw_of(cur_r) - out) + wrap(self.strike_yaw - out)) / 2
            mid = np.array([(cur[0] + F[0]) / 2, (cur[1] + F[1]) / 2, high])
            self.move(mid, palm_down(mid_yaw), hold=0, v_max=V_CARRY, w_max=W_CARRY)
        self.cmd = F.copy()
        if high - F[2] > 0.01:
            self.move(np.array([F[0], F[1], high]), r, hold=4, v_max=V_CARRY, w_max=W_CARRY)
            return self.move(self.cmd, r, n=20, hold=15)
        return self.move(self.cmd, r, hold=15, v_max=V_CARRY, w_max=W_CARRY)

    def align(self, obs, iters=4):
        nail = self.scene["nail"]
        goal = np.array([nail["x"], nail["y"], nail["z"] + STRIKE_H])
        r = self.strike_rot()
        err = None
        for it in range(iters):
            fm = self.measure_face(obs)
            if fm is None or not fm["ok"]:
                self.L("ALIGN face not found", it=it, fm=fm)
                return False
            self.face_f = fm["f"]
            err = fm["w"] - goal
            self.L("ALIGN", it=it, face_w=fm["w"], face_f=fm["f"], err=err, n=fm["n"], n_slab=fm["n_slab"])
            if np.linalg.norm(err[:2]) < ALIGN_TOL and abs(err[2]) < 0.015:
                return True
            if self.rb.remaining is not None and self.rb.remaining < 60:
                self.L("ALIGN out of steps; striking now")
                break
            self.cmd = self.cmd - err
            obs = self.move(self.cmd, r, n=20, hold=12)
        return err is not None and np.linalg.norm(err[:2]) < 2 * ALIGN_TOL

    # ------------------------------------------------------------ STRIKE
    def strike(self):
        rb = self.rb
        if rb.state is None:
            rb.observe()
        r = self.strike_rot()
        low = self.cmd - np.array([0, 0, STRIKE_DZ])
        cur = rb.hold_row()
        rv = continuous_rotvec(r, np.asarray(cur[3:6]))
        hand = cur[6:22]
        zprev = rb.state[2]
        vmin = 0.0
        trace = []
        self.stats["strikes"] += 1
        try:
            for i in range(STRIKE_MAX_ROWS):
                rb.step([row(low, rv, hand)])
                z = rb.state[2]
                v = (z - zprev) / 0.02
                zprev = z
                vmin = min(vmin, v)
                trace.append((round(z, 4), round(v, 3)))
                if vmin < -0.2 and v > -0.05:
                    break
        finally:
            self.L("STRIKE", n=self.stats["strikes"], rows=len(trace), trace=trace, complete=rb.complete)
        return self.move(self.cmd, r, n=25, hold=12)

    def after_strike(self, obs):
        fm = self.measure_face(obs)
        cap = (fm["w"][2] - 0.01) if fm else None
        nail = pc.nail_info(obs, self.scene["board"], exclude_above=cap)
        blk = self.blocking(obs.state)
        self.L("AFTER STRIKE", nail=nail, face=None if fm is None else fm["w"], blocking=blk)
        held = blk["n_blocked"] >= 2 or (fm is not None and fm["ok"])
        if nail is not None:
            old = self.scene["nail"]
            if np.hypot(nail["x"] - old["x"], nail["y"] - old["y"]) < 0.03:
                self.scene["nail"] = {**old, "x": nail["x"], "y": nail["y"], "z": nail["z"]}
        return held

    # ------------------------------------------------------------ RECOVER
    def release_and_clear(self):
        cur = self.rb.hold_row()
        p = np.asarray(cur[0:3])
        r = R.from_rotvec(cur[3:6])
        self.hand_to(HAND_OPEN, n=15, hold=5)
        self.move(p + np.array([0, 0, 0.12]), r, n=30, hold=5)
        self.grasp = None
        self.face_f = None
        self.strike_yaw = None

    # ------------------------------------------------------------ persistence (debug / dev resume)
    def save_state(self, state):
        g = None
        if self.grasp:
            g = {k: (v.as_rotvec().tolist() if k == "rot" else np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in self.grasp.items()}
        d = {"state": state, "scene": self.scene, "fit": self.fit, "grasp": g,
             "face_f": None if self.face_f is None else np.asarray(self.face_f).tolist(),
             "cmd": None if self.cmd is None else np.asarray(self.cmd).tolist(),
             "strike_yaw": self.strike_yaw, "target": self.rb.target, "step": self.rb.step_id}
        try:
            os.makedirs("output", exist_ok=True)
            with open("output/controller_state.json", "w") as f:
                json.dump(d, f, default=str)
        except OSError:
            pass

    def load_state(self, d):
        self.scene = d.get("scene")
        self.fit = d.get("fit")
        g = d.get("grasp")
        if g:
            g = dict(g)
            g["rot"] = R.from_rotvec(g["rot"])
            for k in ("g", "F"):
                g[k] = np.asarray(g[k])
        self.grasp = g
        self.face_f = None if d.get("face_f") is None else np.asarray(d["face_f"])
        self.cmd = None if d.get("cmd") is None else np.asarray(d["cmd"])
        self.strike_yaw = d.get("strike_yaw")
        self.rb.target = d.get("target")

    # ------------------------------------------------------------ main loop
    def run(self, state="SCENE"):
        rb = self.rb
        if rb.state is None:
            rb.observe()
        n_survey = 0
        n_scene_fail = 0
        n_grasp_fail = 0
        n_errors = 0
        strikes_this_grasp = 0
        visits = {}
        while not (rb.ended or rb.complete):
            visits[state] = visits.get(state, 0) + 1
            self.stats["states"].append(state)
            if visits[state] > 40:
                self.L("WATCHDOG: state visited too often", state=state)
                state = "SCENE"
                visits = {}
            nxt = state
            step0 = rb.step_id
            try:
                if state == "SCENE":
                    obs = rb.observe()
                    if self.perceive_scene(obs):
                        nxt = "SURVEY"
                        n_survey = 0
                        n_scene_fail = 0
                    else:
                        n_scene_fail += 1
                        if n_scene_fail % 2 == 1:
                            self.go_view_pose()
                        else:
                            rb.hold(15)
                elif state == "SURVEY":
                    ok = self.survey(n_survey)
                    n_survey += 1
                    if ok:
                        nxt = "GRASP"
                    elif n_survey >= 3:
                        if self.fit is None and self.scene.get("coarse_fit"):
                            self.fit = self.scene["coarse_fit"]
                        nxt = "GRASP" if self.fit is not None else "SCENE"
                elif state == "GRASP":
                    self.stats["grasp_attempts"] += 1
                    if self.do_grasp(n_grasp_fail):
                        nxt = "LIFT"
                    else:
                        n_grasp_fail += 1
                        self.release_and_clear()
                        nxt = "SCENE"
                elif state == "LIFT":
                    if self.do_lift():
                        nxt = "CARRY"
                        strikes_this_grasp = 0
                    else:
                        n_grasp_fail += 1
                        self.release_and_clear()
                        nxt = "SCENE"
                elif state == "CARRY":
                    obs = self.carry()
                    if not self.align(obs):
                        self.L("ALIGN not converged; striking with best estimate")
                    nxt = "STRIKE"
                elif state == "STRIKE":
                    obs = self.strike()
                    strikes_this_grasp += 1
                    if not self.after_strike(obs):
                        self.L("HAMMER LOST")
                        self.release_and_clear()
                        nxt = "SCENE"
                    elif strikes_this_grasp >= MAX_STRIKES_PER_GRASP:
                        self.release_and_clear()
                        nxt = "SCENE"
                    else:
                        self.align(obs, iters=3)
                        nxt = "STRIKE"
            except EpisodeOver:
                raise
            except Exception:
                n_errors += 1
                self.L("ERROR in state", state=state, n_errors=n_errors, tb=traceback.format_exc())
                if n_errors > 12:
                    raise
                try:
                    rb.hold(5)
                except EpisodeOver:
                    raise
                except Exception:
                    pass
                nxt = "SCENE" if state in ("SCENE", "SURVEY", "GRASP", "LIFT") else state
            finally:
                ps = self.stats["phase_steps"]
                ps[state] = ps.get(state, 0) + (rb.step_id or 0) - (step0 or 0)
            if self.stop_after and state == self.stop_after:
                self.L("STOP AFTER", state=state)
                self.save_state(nxt)
                return
            state = nxt
            self.save_state(state)


def main(ctx, stop_after=None, resume=None):
    c = Controller(ctx, stop_after=stop_after)
    try:
        start = "SCENE"
        if resume:
            c.load_state(resume)
            start = resume.get("state") or "SCENE"
            c.L("RESUME", state=start)
        c.run(start)
    except EpisodeOver:
        c.L("EPISODE OVER", complete=c.rb.complete, ended=c.rb.ended)
    except Exception:
        c.L("FATAL", tb=traceback.format_exc())
    finally:
        c.stats.update({"complete": c.rb.complete, "ended": c.rb.ended, "step": c.rb.step_id,
                        "calls": c.rb.n_calls})
        c.stats["states"] = c.stats["states"][-60:]
        try:
            os.makedirs("output", exist_ok=True)
            with open("output/controller_summary.json", "w") as f:
                json.dump(c.stats, f, default=str)
        except OSError:
            pass
        c.L("SUMMARY", stats=c.stats)
    return c
