"""PegInsertionSide-v1 controller (pd_ee_delta_pose), vision-only feedback.

Pipeline
 0. t=0: peg from base (+wrist) camera, box entrance face + hole from base/wrist camera
    (model-based tunnel fit, sub-mm).
 1. Approach above the grasp point (tail side of the peg centre), rotating to the peg yaw concurrently;
    refine the peg estimate from the wrist camera on the way.
 2. Descend, close.  Fingers closed on nothing -> regrasp(): rise, re-detect the peg, grasp again.
 3. Lift; every step measure the peg pose in the hand (pitch sag, yaw, height, axial slip) from the wrist
    camera and servo the TCP so the *peg axis* lines up with the hole axis, head a few cm before the face.
    Payload sag of the TCP (peg weight vs the joint PD) is cancelled by an integral z feedforward (sag_ff).
 4. Insert along the hole axis with the in-hand estimate frozen at the aligned pre-insert pose; stall
    detection -> back off, re-measure, retry.
"""
import os
import json
import time
import numpy as np

import robot_api as robot
from geom import rotvec, rotvec_to_R
from motion import Servo, tcp_pose, gripper_width, topdown_R, reach_excess, BASE_XY
from perception import (detect_peg_base, detect_peg_hand, detect_hole, refine_hole, peg_in_hand)

HORIZON = 100
G_OFF = 0.05          # grasp point distance behind peg centre (toward tail)
Z_PRE = 0.08          # pre-grasp TCP height
STAGE_D = 0.035       # head standoff in front of the face for the aligned pre-insert pose
INS_DS = 0.010        # insertion target advance per step (head outside / entering the hole)
INS_DS_IN = 0.015     # ... once the head is >2 cm inside (the tunnel guides the peg)
LOOK_Y = 0.07         # sideways TCP offset of the active hole view (hole left of the peg in the image)
LOOK_D = 0.12         # pre-grasp look: camera distance in front of the face
LOOK_H = 0.19         # pre-grasp look: camera height above the hole
C_OFF = np.array([0.0465, -0.02, -0.0674])   # wrist camera position in the TCP frame
KI_Z = 0.5            # payload-sag integrator gain on world z (sag_ff)
LOG_DIR = '/workspace/data/runs'
SNAP = True           # save camera frames at phase boundaries (debug)


def _f(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, dict):
        return {k: _f(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_f(v) for v in x]
    return x


class Runner:
    def __init__(self):
        self.obs = robot.observe()
        self.cams = robot.cameras()
        self.done = False
        self.succ = False
        self.servo = Servo(kp=2.0, kd=1.5, kr=2.0)
        self.log = []
        self.info = {}
        self.t0 = time.time()

    # ------------------------------------------------------------------ utils
    @property
    def n(self):
        return int(self.obs.get('native_step', 0))

    def left(self):
        return HORIZON - self.n

    def note(self, **kw):
        self.log.append({k: _f(v) for k, v in kw.items()})

    def step(self, a, phase):
        if self.done:
            return False
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        res = robot.step([float(v) for v in a])
        self.obs = res['observation']
        self.succ = bool(res.get('native_success', False))
        term = bool(res.get('terminal', False))
        if self.succ or term or self.n >= HORIZON:
            self.done = True
        else:
            self.cams = robot.cameras()
        p, R = tcp_pose(self.obs)
        self.note(n=self.n, phase=phase, a=a, p=p, rv=rotvec(R), w=gripper_width(self.obs), succ=self.succ)
        return not self.done

    def snap(self, tag):
        if not SNAP:
            return
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            ep = self.obs.get('episode', 'x')
            np.savez_compressed(os.path.join(LOG_DIR, 'ep%s_%s.npz' % (ep, tag)), **self.cams)
            json.dump(self.obs, open(os.path.join(LOG_DIR, 'ep%s_%s_obs.json' % (ep, tag)), 'w'))
        except Exception:
            pass

    def save_log(self):
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            ep = self.obs.get('episode', 'x')
            fn = os.path.join(LOG_DIR, 'ep%s_%d.json' % (ep, int(time.time())))
            json.dump(dict(info={k: _f(v) for k, v in self.info.items()}, log=self.log), open(fn, 'w'))
        except Exception:
            pass

    # ------------------------------------------------------------ perception
    def hole_estimates(self, cams_list=('hand_camera', 'base_camera'), h0=None):
        out = []
        for cam in cams_list:
            try:
                det = detect_hole(self.obs, self.cams, cam, min_behind=4 if cam == 'base_camera' else 8)
                if det is None:
                    continue
                if self.face is None or cam == 'base_camera' or self.face_cam == 'hand_camera':
                    self.face, self.face_cam = det, cam   # face plane is useful even without a hole
                if det.get('hole') is None:
                    continue
                rf = refine_hole(self.obs, self.cams, det, cam, h0=h0, L=self.L, fit_h=True)
                if rf is None or rf['agree'] < 0.95:
                    continue
                trunc = any(det['touches'].values())
                nb = det['npx']                      # pixels seen through the opening
                if cam == 'hand_camera':
                    w = 3.0 * min(1.0, nb / 300.0) * (0.3 if trunc else 1.0)
                    good = (not trunc) and nb >= 100
                else:
                    w = 0.5 * min(1.0, nb / 20.0)
                    good = nb >= 15
                out.append(dict(cam=cam, hole=rf['hole'], xb=rf['xb'], h=rf['h'], w=w, npx=rf['npx'], nb=nb,
                                trunc=trunc, good=good, agree=rf['agree'], ztop=det['face_ztop'], n=self.n))
            except Exception as e:  # perception must never kill the episode
                self.note(err='hole:%s:%r' % (cam, e))
        return out

    def good_hand_holes(self):
        return [h for h in self.holes if h['cam'] == 'hand_camera' and h['good']]

    def cross_checked(self):
        """Large (but truncated/occluded) wrist fits confirmed by an independent good base-camera fit."""
        hand = [h for h in self.holes if h['cam'] == 'hand_camera' and h['nb'] >= 50]
        base = [h for h in self.holes if h['cam'] == 'base_camera' and h['good']]
        ok = [a for a in hand if any(np.linalg.norm(a['hole'] - b['hole']) < 0.0015 for b in base)]
        if not ok:
            return []
        return ok + [b for b in base if any(np.linalg.norm(a['hole'] - b['hole']) < 0.0015 for a in ok)]

    def hole_confident(self):
        gh = self.good_hand_holes()
        if gh:
            # a single untruncated wrist fit is trusted unless it contradicts every good base fit by >3 mm (ep12: a
            # truncated wrist fit was 6 mm off with a perfect model agreement score; base fits with >=15 px were all
            # within 0.8 mm, ep4-ep13).  Two agreeing wrist fits override the base camera.
            gb = [h for h in self.holes if h['cam'] == 'base_camera' and h['good']]
            pair = any(np.linalg.norm(a['hole'] - b['hole']) < 0.0015 for i, a in enumerate(gh) for b in gh[i + 1:])
            conflict = bool(gb) and all(min(np.linalg.norm(a['hole'] - b['hole']) for b in gb) > 0.003 for a in gh)
            if pair or not conflict:
                return True
        return bool(self.cross_checked())

    def fuse_holes(self):
        if not self.holes:
            return None
        # quality tiers: unoccluded full wrist views > cross-checked wrist+base > any good view > everything
        H = self.good_hand_holes() or self.cross_checked() or [h for h in self.holes if h['good']] or self.holes
        P = np.array([h['hole'] for h in H]); W = np.array([h['w'] for h in H]) + 1e-6
        X = np.array([h['xb'] for h in H])
        med = np.median(P, 0)
        keep = np.linalg.norm(P - med, axis=1) < 0.004
        if keep.sum() == 0:
            keep[:] = True
        hole = (P[keep] * W[keep, None]).sum(0) / W[keep].sum()
        xb = (X[keep] * W[keep, None]).sum(0) / W[keep].sum()
        xb[2] = 0.0
        xb /= np.linalg.norm(xb)
        hh = np.array([h['h'] for h in H])[keep]
        return hole, xb, float(np.median(hh))

    def update_holes(self, cams_list):
        new = self.hole_estimates(cams_list)
        if new:
            self.holes += new
            f = self.fuse_holes()
            if f is not None:
                self.hole, self.xb, self.h = f
            self.update_L()
        return new

    def update_L(self):
        """Box half-size == peg half-length: top of the entrance face seen by the base camera = 2L;
        fallback: peg length seen by the wrist camera during the approach."""
        ztops = [h['ztop'] for h in self.holes if h['cam'] == 'base_camera']
        if self.face is not None and self.face.get('face_ztop') is not None and self.face_cam == 'base_camera':
            ztops.append(self.face['face_ztop'])
        if ztops:
            L_box = float(np.median(ztops) / 2.0 + 0.0005)
            if 0.08 < L_box < 0.13:
                self.L = L_box
                return
        if self.peg is not None and self.peg.get('L_hand') is not None:
            self.L = float(np.clip(self.peg['L_hand'], 0.08, 0.13))

    # ------------------------------------------------------------ phases
    def perceive_initial(self):
        self.L = 0.105
        self.r = 0.02
        self.widths = []
        self.face = None
        self.face_cam = None
        pb = detect_peg_base(self.obs, self.cams)
        ph = None
        try:
            ph = detect_peg_hand(self.obs, self.cams)
        except Exception:
            ph = None
        self.peg = None
        if pb is not None:
            self.peg = dict(c=np.array(pb['center']), yaw=pb['yaw'], L=pb['L'], r=pb['r'])
            self.L = float(np.clip(pb['L'], 0.08, 0.13))
            self.r = float(np.clip(pb['r'], 0.014, 0.026))
        if ph is not None and ph.get('center') is not None and ph['width'] > 0.025 and ph['ntop'] > 300:
            if self.peg is None or np.linalg.norm(ph['center'] - self.peg['c']) < 0.03:
                self.peg = dict(c=np.array(ph['center']), yaw=ph['yaw'], L=self.L, r=self.r)
        self.holes = self.hole_estimates()
        self.update_L()
        f = self.fuse_holes()
        self.info.update(peg0=self.peg, L0=self.L, r0=self.r, holes0=[{k: _f(v) for k, v in h.items()} for h in self.holes])
        if f is not None:
            self.hole, self.xb, self.h = f
            if any(h['good'] for h in self.holes):
                self.r = float(np.clip(self.h - 0.003, 0.014, 0.026))
        else:
            self.hole, self.xb, self.h = None, None, None
        print('[init] peg', None if self.peg is None else (np.round(self.peg['c'], 4), round(self.peg['yaw'], 3)),
              'L %.4f r %.4f' % (self.L, self.r), 'hole', None if self.hole is None else np.round(self.hole, 4),
              'xb', None if self.xb is None else np.round(self.xb, 4), 'nest', len(self.holes))

    def set_r(self):
        """Peg half-width: wrist top-face width (direct, ~0.3 mm); fallback hole half-size - 0.003 from a
        good hole fit."""
        if self.widths:
            self.r = float(np.clip(0.5 * np.median(self.widths), 0.014, 0.026))
        else:
            good = [h['h'] for h in self.holes if h['good']]
            if good:
                self.r = float(np.clip(np.median(good) - 0.003, 0.014, 0.026))
        print('[r] widths', np.round(self.widths, 4), 'h', self.h, '-> r %.4f' % self.r)

    def grasp_target(self):
        yaw = self.peg['yaw']
        ax = np.array([np.cos(yaw), np.sin(yaw)])
        g_off = G_OFF
        while g_off > 0.0 and reach_excess(np.r_[self.peg['c'] - g_off * ax, 0.02], 0.025) > 0:
            g_off -= 0.01         # far-out peg with the tail pointing away: grasp nearer the centre
        gxy = self.peg['c'] - g_off * ax
        return gxy, topdown_R(yaw)

    def approach_and_grasp(self):
        if self.peg is None:
            # look around: rise to see the table from the wrist (rare)
            for _ in range(4):
                p, R = tcp_pose(self.obs)
                a, _, _ = self.servo.action(self.obs, np.array([0.0, -0.15, 0.25]), R, 1.0)
                self.step(a, 'search')
                pb = detect_peg_base(self.obs, self.cams)
                if pb is not None:
                    self.peg = dict(c=np.array(pb['center']), yaw=pb['yaw'], L=pb['L'], r=pb['r'])
                    break
            if self.peg is None:
                return False
        self.servo.reset()
        nref = 0
        nconv = 0
        for i in range(40):
            gxy, Rg = self.grasp_target()
            tgt = np.array([gxy[0], gxy[1], Z_PRE])
            a, ep, er = self.servo.action(self.obs, tgt, Rg, 1.0)
            # descend may start before the yaw has fully converged (rotation is the slow axis, ~0.037 rad/step);
            # open fingers (+-4 cm, pads +-1 cm along TCP x): at 0.4 rad yaw error a 2.5 cm half-width peg spans
            # 0.025/cos + 0.01*tan = 3.1 cm (+4 mm xy error) -> still clears; the descend loop waits for er < 0.05
            if np.linalg.norm(ep[:2]) < 0.004 and abs(ep[2]) < 0.012 and np.linalg.norm(er) < 0.4:
                nconv += 1
                if nref >= 1 or nconv > 2:
                    break
            if not self.step(a, 'approach'):
                return False
            p, R = tcp_pose(self.obs)
            nb = sum(1 for h in self.holes if h['cam'] == 'base_camera')
            if i % 3 == 2 and nb < 3:
                self.update_holes(('base_camera',))
            # wrist refinement once the camera is above the peg region
            if p[2] > 0.065 and np.linalg.norm(p[:2] - gxy) < 0.06:
                try:
                    ph = detect_peg_hand(self.obs, self.cams)
                except Exception:
                    ph = None
                if ph is not None and ph.get('center') is not None and 0.026 < ph['width'] < 0.06 and ph['ntop'] > 300:
                    if np.linalg.norm(ph['center'] - self.peg['c']) < 0.04:
                        dyaw = (ph['yaw'] - self.peg['yaw'] + np.pi) % (2 * np.pi) - np.pi
                        if abs(dyaw) < 0.5:
                            self.peg['c'] = np.array(ph['center'])
                            self.peg['yaw'] = float(ph['yaw'])
                            self.peg['width'] = ph['width']
                            self.widths.append(ph['width'])
                            if ph['smax'] - ph['smin'] > 0.16 and abs(ph['smax'] - ph['smin'] - 2 * self.L) < 0.02:
                                self.peg['L_hand'] = 0.5 * (ph['smax'] - ph['smin'])
                            nref += 1
            print('[approach]', self.n, 'ep %.4f er %.3f' % (np.linalg.norm(ep), np.linalg.norm(er)), 'peg', np.round(self.peg['c'], 4), round(self.peg['yaw'], 3), 'nref', nref)
        self.info['peg_ref'] = dict(self.peg)
        self.set_r()
        # descend
        gxy, Rg = self.grasp_target()
        zg = float(np.clip(self.r + 0.0005, 0.013, 0.03))
        tgt = np.array([gxy[0], gxy[1], zg])
        for i in range(18):
            a, ep, er = self.servo.action(self.obs, tgt, Rg, 1.0)
            if np.linalg.norm(ep) < 0.003 and np.linalg.norm(er) < 0.05:
                break
            if i >= 12 and np.linalg.norm(ep) < 0.008 and np.linalg.norm(er) < 0.08:
                break
            if not self.step(a, 'descend'):
                return False
            print('[descend]', self.n, 'ep %.4f er %.3f' % (np.linalg.norm(ep), np.linalg.norm(er)))
        # close
        wprev = gripper_width(self.obs)
        for i in range(4):
            a, ep, er = self.servo.action(self.obs, tgt, Rg, -1.0)
            if not self.step(a, 'close'):
                return False
            w = gripper_width(self.obs)
            print('[close]', self.n, 'w %.4f' % w)
            if i >= 1 and abs(w - wprev) < 0.0015:
                break
            wprev = w
        w = gripper_width(self.obs)
        self.info['grip_w'] = w
        self.snap('grasped')
        return w > 0.015

    def regrasp(self):
        """Missed grasp (fingers closed on nothing): open, rise so the wrist camera overlooks the table,
        re-detect the peg without a proximity gate (it may have been pushed), then grasp again."""
        p, R = tcp_pose(self.obs)
        up = p.copy(); up[2] = 0.15
        self.servo.reset()
        for i in range(5):
            a, ep, er = self.servo.action(self.obs, up, R, 1.0)
            if not self.step(a, 'regrasp'):
                return False
            if i >= 2 and np.linalg.norm(ep) < 0.03:
                break
        cand = []
        try:
            ph = detect_peg_hand(self.obs, self.cams)
            if ph is not None and ph.get('center') is not None and 0.025 < ph['width'] < 0.06 and ph['ntop'] > 200:
                cand.append(dict(c=np.array(ph['center']), yaw=float(ph['yaw']), L=self.L, r=self.r))
        except Exception as e:
            self.note(err='regrasp_hand:%r' % e)
        try:
            pb = detect_peg_base(self.obs, self.cams)
            if pb is not None:
                cand.append(dict(c=np.array(pb['center']), yaw=pb['yaw'], L=self.L, r=self.r))
        except Exception as e:
            self.note(err='regrasp_base:%r' % e)
        print('[regrasp]', self.n, [(np.round(c['c'], 4), round(c['yaw'], 3)) for c in cand])
        if not cand:
            return False
        self.peg = cand[0]
        return self.approach_and_grasp()

    def inhand(self):
        p, R = tcp_pose(self.obs)
        try:
            ih = peg_in_hand(self.obs, self.cams, p, R, self.r, self.L)
        except Exception as e:
            self.note(err='inhand:%r' % e)
            return None
        if ih is None or ih.get('head') is None or ih['n'] < 400:
            return None
        if abs(ih['theta']) > 0.35 or abs(ih['psi']) > 0.2:
            return None
        return ih

    def insertion_pose(self, ih, s):
        """TCP pose so that peg axis == xb and head centre == hole + s*xb."""
        yaw_b = np.arctan2(self.xb[1], self.xb[0])
        R0 = topdown_R(yaw_b)
        a = ih['axis']
        # minimal rotation taking in-hand axis a to e_x
        ex = np.array([1.0, 0, 0])
        v = np.cross(a, ex)
        sn, cs = np.linalg.norm(v), np.dot(a, ex)
        Rc = np.eye(3) if sn < 1e-9 else rotvec_to_R(v / sn * np.arctan2(sn, cs))
        R_t = R0 @ Rc
        p_t = self.hole + s * self.xb - R_t @ ih['head']
        return p_t, R_t

    def stage_s(self, ih):
        """Head standoff for the aligned pre-insert pose: STAGE_D, reduced if that TCP pose is out of reach
        (box yawed so the entrance faces away from the robot)."""
        s = -STAGE_D
        while s < -0.006:
            p_t, _ = self.insertion_pose(ih, s)
            if reach_excess(p_t, 0.025) <= 0:
                break
            s += 0.005
        return s

    def sag_ff(self, p_t, learn=True):
        """Payload sag compensation: integral action on world z.  Holding the peg, the P-only servo settles
        0.4-1.9 mm BELOW its target (peg weight against the joint PD; constant through the insertion, ep4-ep17),
        i.e. the head enters low, eating up to 2/3 of the 3 mm clearance.  z is always across the hole axis, so
        this never interferes with the axial lag logic.  Learns only near rest (no windup on arrival)."""
        p, _ = tcp_pose(self.obs)
        ez = p_t[2] - p[2]
        vz = 0.0 if self.pz_prev is None else p[2] - self.pz_prev
        self.pz_prev = p[2]
        if learn and abs(ez) < 0.004 and abs(vz) < 0.002:
            self.iz = float(np.clip(self.iz + KI_Z * ez, -0.002, 0.006))
        return np.array([0.0, 0.0, self.iz])

    def hole_guess(self):
        """Best available hole centre + insertion axis (coarse if needed: face centre at box mid-height)."""
        if self.hole is not None:
            return self.hole.copy(), self.xb.copy()
        f = self.face
        if f is None:   # nothing seen: centre of the box spawn region, entrance face toward the robot side
            return np.array([0.0, 0.3 - self.L, self.L]), np.array([0.0, 1.0, 0.0])
        g = f['face_pt'] + f['t'] * 0.5 * (f['face_u'][0] + f['face_u'][1])
        g[2] = self.L
        xb = f['xb'].copy()
        return g, xb

    def look_cam(self, g, xb, Rs):
        """Wrist-camera position for a downward view of the opening: in front of the face (~32 deg off
        nadir, ~58 deg off the face normal), shifted along the face toward the robot base when the nominal
        spot is outside the arm's reach."""
        n_out = -xb
        t = np.cross(np.array([0.0, 0.0, 1.0]), n_out)
        sgn = -1.0 if t[:2] @ (g[:2] - BASE_XY) > 0 else 1.0
        best = None
        for (D, S, H) in [(LOOK_D, 0.0, LOOK_H), (LOOK_D, 0.03, LOOK_H), (0.11, 0.05, 0.18), (0.10, 0.06, 0.17)]:
            cam = g + D * n_out + sgn * S * t
            cam[2] = g[2] + H
            exc = max(reach_excess(cam - R @ C_OFF, 0.03) for R in Rs)
            if best is None or exc < best[0]:
                best = (exc, cam)
            if exc <= 0:
                break
        return best[1]

    def look_pregrasp(self, max_steps=10):
        """Before grasping (no peg to occlude the wrist view): put the wrist camera ~12 cm in front of and
        ~19 cm above the (guessed) hole, looking down; the hole appears ~32 deg off-axis, upper image half
        for any TCP yaw within +-90 deg of the box axis.  Meanwhile keep turning toward the grasp yaw."""
        if self.peg is None:
            return False
        _, Rg = self.grasp_target()
        self.servo.reset()
        near = 0
        for i in range(max_steps):
            g, xb = self.hole_guess()          # re-aimed as soon as a first (coarse) face/hole fit exists
            p, R = tcp_pose(self.obs)
            cam = self.look_cam(g, xb, (R, Rg))
            p_t = cam - R @ C_OFF
            p_t[2] = max(p_t[2], 0.12)
            a, ep, er = self.servo.action(self.obs, p_t, Rg, 1.0)
            if not self.step(a, 'look0'):
                return False
            p, R = tcp_pose(self.obs)
            if np.linalg.norm(p - p_t) < 0.05:
                near += 1
                new = self.update_holes(('hand_camera',))
                print('[look0]', self.n, 'ep %.4f' % np.linalg.norm(p - p_t), 'new',
                      [(np.round(h['hole'], 4).tolist(), round(h['h'], 4), h['nb'], h['trunc']) for h in new])
                gh = self.good_hand_holes()
                if len(gh) >= 2 and np.linalg.norm(gh[-1]['hole'] - gh[-2]['hole']) < 0.0015:
                    break
                if near >= 4 and self.hole_confident():
                    break
        self.snap('look0')
        return self.hole_confident()

    def active_look(self, max_steps=14):
        """Move the wrist camera to a view of the entrance face: TCP yaw = box yaw (needed for insertion
        anyway), TCP ~17 cm in front of the face, high, and offset sideways so the hole projects into the
        image half not covered by the grasped peg.  Collect wrist hole fits until two agree."""
        g, xb = self.hole_guess()
        if g is None:
            return False
        head_x = self.ih['head'][0] if self.ih is not None else G_OFF + self.L
        D = max(0.17, head_x + 0.03)
        H = (D - 0.0465) / np.tan(np.radians(33.0)) - 0.0674
        R_look = topdown_R(np.arctan2(xb[1], xb[0]))
        p_look = g - D * xb + LOOK_Y * R_look[:, 1]
        p_look[2] = g[2] + H
        for _ in range(20):                    # pull toward the base until reachable
            if reach_excess(p_look, 0.03) <= 0:
                break
            u = np.r_[p_look[:2] - BASE_XY, 0.0]
            p_look -= 0.01 * u / np.linalg.norm(u)
        self.servo.reset()
        for i in range(max_steps):
            a, ep, er = self.servo.action(self.obs, p_look, R_look, -1.0)
            if not self.step(a, 'look'):
                return False
            p, R = tcp_pose(self.obs)
            if np.linalg.norm(p - p_look) < 0.03:
                new = self.update_holes(('hand_camera',))
                print('[look]', self.n, 'ep %.4f' % np.linalg.norm(p - p_look), 'new',
                      [(np.round(h['hole'], 4).tolist(), round(h['h'], 4), h['nb'], h['trunc']) for h in new])
                gh = self.good_hand_holes()
                if len(gh) >= 2 and np.linalg.norm(gh[-1]['hole'] - gh[-2]['hole']) < 0.0015:
                    break
            # the camera sees the grasped peg too: keep the in-hand estimate fresh
            ihn = self.inhand()
            if ihn is not None:
                self.ih = self.ih_prev = ihn
        self.snap('look')
        return True

    def transport(self):
        self.servo.reset()
        p, R = tcp_pose(self.obs)
        # sag prior ~ peg mass (8 L r^2, density 1000): the insert-phase disturbance observer gave 3.7-7.4 mm/kg of
        # commanded z (median 5.6, ep4-ep18) -> slightly low 5.0 (over-compensation would lift the head)
        self.iz = float(np.clip(5.0 * 8.0 * self.L * self.r ** 2, 0.0008, 0.0035))
        self.pz_prev = None
        ih = self.inhand()
        self.ih0 = ih
        self.ih = ih
        self.ih_prev = ih
        if ih is not None:
            self.info['ih_table'] = dict(theta=ih['theta'], psi=ih['psi'], head=ih['head'], xb=ih['xb'])
        # lift straight up a bit, already turning toward the box yaw (rotation is the slow axis)
        lift = p.copy(); lift[2] = max(p[2] + 0.05, 0.07)
        _, xb = self.hole_guess()
        R_lift = R if xb is None else topdown_R(np.arctan2(xb[1], xb[0]))
        for i in range(2):
            a, ep, er = self.servo.action(self.obs, lift, R_lift, -1.0)
            if not self.step(a, 'lift'):
                return False
        if not self.hole_confident():
            self.active_look()
            self.servo.reset()
        if self.hole is None:
            print('[transport] no hole estimate after active look')
            return False
        best_hole_n = self.n
        n_ih = 0
        still = 0
        p_prev = None
        for i in range(60):
            ihn = self.inhand()
            if ihn is not None:
                n_ih += 1
                if self.ih is None or self.ih_prev is None:
                    self.ih = ihn
                else:   # smoothing: light while the sag settles after the lift, then heavier (frame noise
                        # ~0.3-0.5 deg in theta/psi otherwise jitters the target rotation by ~0.01 rad, ep9)
                    k = 0.6 if n_ih <= 3 else 0.35
                    self.ih = dict(ihn)
                    for key in ('head', 'axis'):
                        self.ih[key] = k * ihn[key] + (1 - k) * self.ih_prev[key]
                    self.ih['axis'] = self.ih['axis'] / np.linalg.norm(self.ih['axis'])
                self.ih_prev = self.ih
            if self.ih is None:
                ih_use = dict(head=np.array([G_OFF + self.L, 0, 0.0]), axis=np.array([1.0, 0, 0]))
            else:
                ih_use = self.ih
                self.ih_prev = self.ih
            p_t, R_t = self.insertion_pose(ih_use, self.stage_s(ih_use))
            p, R = tcp_pose(self.obs)
            # extra standoff while the rotation converges, faded out smoothly with the rotation error (a switch at
            # er 0.15 stepped the target by 4 cm; the sudden translation kicked the yaw ~0.5 deg past, ep12)
            er_n = np.linalg.norm(rotvec(R_t @ R.T))
            p_t = p_t - 0.04 * float(np.clip((er_n - 0.03) / 0.2, 0.0, 1.0)) * self.xb
            # learn the sag only within 1 cm of the target: the z lag of a fast horizontal move is not sag
            # (ep18 step 29: |ep| 6 cm, ez 3.5 mm -> iz wound up 0.9 -> 2.7 mm; true value 1.1-1.2 mm)
            a, ep, er = self.servo.action(self.obs, p_t, R_t, -1.0, ff=self.sag_ff(p_t, learn=np.linalg.norm(p_t - p) < 0.01))
            conv = np.linalg.norm(ep) < 0.002 and np.linalg.norm(er) < 0.008
            # at rest close to the target: nothing more to gain (ep17 hovered at |ep| 2.0-2.1 mm for 5 steps)
            near = np.linalg.norm(ep) < 0.0035 and np.linalg.norm(er) < 0.010
            still = still + 1 if (near and p_prev is not None and np.linalg.norm(p - p_prev) < 0.0007) else 0
            p_prev = p.copy()
            if (conv and i >= 3) or still >= 4 or \
                    (self.left() < 22 and np.linalg.norm(ep) < 0.006 and np.linalg.norm(er) < 0.02) or self.left() < 16:
                break
            if not self.step(a, 'stage'):
                return False
            # opportunistic hole re-estimate from the base camera (arm moved away from the t=0 occlusion)
            if self.n - best_hole_n >= 6:
                best_hole_n = self.n
                self.update_holes(('base_camera',))
            if self.ih is not None:
                print('[stage]', self.n, 'ep %.4f er %.4f ez %.4f iz %.4f' % (np.linalg.norm(ep), np.linalg.norm(er), ep[2], self.iz),
                      'theta %.2f psi %.2f head %s xb %.4f' % (np.degrees(self.ih['theta']), np.degrees(self.ih['psi']),
                                                              np.round(self.ih['head'], 4), self.ih['xb']))
        self.info['ih_stage'] = None if self.ih is None else dict(theta=self.ih['theta'], psi=self.ih['psi'], head=self.ih['head'], xb=self.ih['xb'])
        self.info['iz_stage'] = self.iz
        self.info['hole'] = dict(hole=self.hole, xb=self.xb, h=self.h)
        self.snap('staged')
        return True

    def insert(self):
        ih = self.ih if self.ih is not None else dict(head=np.array([G_OFF + self.L, 0, 0.0]), axis=np.array([1.0, 0, 0]), xb=G_OFF)
        xb0 = ih.get('xb', G_OFF)
        s = self.stage_s(ih)
        s_max = self.L + 0.01
        blocked = 0
        retries = 0
        self.servo.reset()
        while not self.done:
            p_t, R_t = self.insertion_pose(ih, s)
            p, R = tcp_pose(self.obs)
            head_w = p + R @ ih['head']
            along = (head_w - self.hole) @ self.xb
            lag = s - along
            # advance the reference only while the peg follows
            ds = INS_DS_IN if along > 0.02 else INS_DS
            if lag < 0.006 + 0.5 * (ds - INS_DS):
                if s >= s_max - 1e-6 and s_max < self.L + 0.04:
                    s_max += 0.01         # reached the nominal depth without success: push deeper
                s = min(s + ds, s_max)
                blocked = 0
            else:
                blocked += 1
            if blocked >= 3 and retries < 3:
                # back off, re-measure in-hand pose at rest, retry
                retries += 1
                print('[insert] blocked at along %.4f, retry %d' % (along, retries))
                self.snap('blocked%d' % retries)
                s = min(along, -0.01) - 0.01
                for j in range(3):
                    p_t, R_t = self.insertion_pose(ih, s)
                    a, ep, er = self.servo.action(self.obs, p_t, R_t, -1.0, kd=0.0, ff=self.sag_ff(p_t, learn=False))
                    if not self.step(a, 'backoff'):
                        return
                ihn = self.inhand()
                if ihn is not None:
                    ih = ihn
                    if ih.get('xb') is not None:
                        xb0 = ih['xb']      # slip reference follows the re-measured grasp
                blocked = 0
                continue
            p_t, R_t = self.insertion_pose(ih, s)
            # sag integrator keeps learning until the head reaches the face, then holds (no lateral wind-up in contact)
            a, ep, er = self.servo.action(self.obs, p_t, R_t, -1.0, kd=0.0, ff=self.sag_ff(p_t, learn=along < -0.003))
            if not self.step(a, 'insert'):
                return
            ihn = self.inhand()
            slip = None if ihn is None or ihn.get('xb') is None else ihn['xb'] - xb0
            if slip is not None and slip < -0.004 and ihn['n_white'] > 50:
                blocked += 1      # peg pushed back through the fingers: head is blocked
            print('[insert]', self.n, 's %.4f along %.4f lag %.4f ep %.4f er %.4f ez %.4f iz %.4f slip %s' % (
                s, along, lag, np.linalg.norm(ep), np.linalg.norm(er), ep[2], self.iz, None if slip is None else round(slip, 4)))

    def run(self):
        try:
            self.perceive_initial()
            if self.hole is None:
                print('no hole estimate at t=0')
            if not self.hole_confident():
                self.look_pregrasp()
            ok = self.approach_and_grasp()
            tries = 1
            while not ok and not self.done and tries < 3 and self.left() > 40:
                tries += 1
                ok = self.regrasp()
            print('[grasp] ok', ok, 'step', self.n, 'tries', tries)
            if ok and not self.done:
                if self.hole is None:
                    self.update_holes(('hand_camera', 'base_camera'))
                if self.transport():
                    self.insert()
        finally:
            self.info['success'] = self.succ
            self.info['final_step'] = self.n
            self.save_log()
            print('[done] success', self.succ, 'step', self.n, 'wall %.1fs' % (time.time() - self.t0))


def main():
    Runner().run()


if __name__ == '__main__':
    main()
