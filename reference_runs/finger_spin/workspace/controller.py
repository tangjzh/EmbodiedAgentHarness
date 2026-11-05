"""controller.py (ctrl3): full-speed bang-bang strokes of the fingertip on the LEFT of the spinner,
timed by bar phase p = th mod 180 (deg) and CW speed.  Uses only the observation.
Up-stroke (u1=+1) pushes the left arm's lower (trailing) face upward => CW (rewarded) torque.
Down-stroke (u1=-1) must finish before the NEXT arm reaches p_block (it would be hit head-on).
psi = fingertip angle about proximal joint B; q2 held so fingertip radius about B = rB."""
import numpy as np

B = np.array([-0.2, 0.4]); L1 = 0.18; S = 0.1455; D2R = np.pi / 180

DEFAULT = dict(
    rB=0.28,            # fingertip radius about proximal joint (held via q2) in sustain regime
    p_clear=85.0,       # up-stroke should reach the arm (psi~-18) when it is at p <= p_clear
    p_block=165.0,      # bar phase at which the NEXT arm starts blocking the return path
    T_up=0.03,          # up-stroke starts this long (s) before the arm reaches p_clear
    T0=0.02, k_v=0.002, v_ret=500.0, psi_clear=-18.0, w_infl=1.15,  # return needs T0+k_v*dq1+(psi-psi_clear)/v_ret
    rB_ret=0.25,        # paddle-wheel: retract fingertip (outside sweep circle) during the return stroke
    psi_bot=-25.0, psi_top=0.0,
    kp1=6.0, kd1=0.08, kp2=5.0, kd2=0.05,
    w_fast=5.0,         # CW rad/s above which time-based commutation is used
    slow_lo=40.0, slow_hi=95.0,   # slow regime: full-speed strike when p in [slow_lo, slow_hi]
    ccw_lo=15.0, ccw_hi=100.0,    # CCW-spinning regime: head-on strike window
    psi_top_slow=-2.0, rB_slow=0.30, psi_top_wig=15.0,
    stall_ticks=3, wig_period=7, w_stall=1.5,
)
MODES = {'down': 0, 'up': 1, 'slow_up': 2, 'slow_down': 3, 'wiggle_up': 4, 'wiggle_down': 5, 'pre': 6}


def q2_for_rB(rB):
    c = (rB ** 2 - L1 ** 2 - S ** 2) / (2 * L1 * S)
    return float(np.arccos(np.clip(c, -1, 1)))


def psi_of(q):
    a2 = q[0] + q[1]
    x = L1 * np.cos(q[0]) + S * np.cos(a2)
    z = L1 * np.sin(q[0]) + S * np.sin(a2)
    return float(np.arctan2(z, x))


def q1_for_psi(psi, q2):
    return psi - np.arctan2(S * np.sin(q2), L1 + S * np.cos(q2))


def decode(obs):
    if isinstance(obs, dict):
        pos = np.asarray(obs['position'], float); vel = np.asarray(obs['velocity'], float)
    else:                                   # flat [position(4), velocity(3), touch(2)] fallback
        flat = np.asarray(obs, float).ravel(); pos, vel = flat[:4], flat[4:7]
    return pos[:2], vel[:2], float(np.arctan2(-pos[2], pos[3])), float(vel[2])


def act(obs, mem, P):
    q, dq, th, w = decode(obs)
    psi = psi_of(q) / D2R
    p = (th / D2R) % 180.0
    wc = -w                      # CW speed rad/s (rewarded direction positive)
    wd = wc / D2R                # deg/s
    info = dict(psi=psi, p=p, s=np.nan, s_end=np.nan)
    mem['stall'] = mem.get('stall', 0) + 1 if abs(w) < P['w_stall'] else 0
    rB = P['rB']; top = P['psi_top']
    if wc > P['w_fast']:
        phi = (P['p_clear'] - p) % 180.0                  # deg since last arm passed p_clear
        phi_blk = 180.0 - (P['p_block'] - P['p_clear'])   # phi at which next arm reaches p_block
        if phi < mem.get('phi_prev', phi) - 90.0:          # a new arm passed p_clear: new cycle
            mem['latch'] = False
        mem['phi_prev'] = phi
        pre = phi >= 180.0 - wd * P['T_up']
        if pre:
            mem['latch'] = False; up = True; mode = 'pre'
        elif phi >= phi_blk or mem.get('latch', False):
            up = False; mode = 'down'
        else:
            need = P['T0'] + P['k_v'] * max(dq[0], 0.0) + max(0.0, psi - P['psi_clear']) / P['v_ret']
            tb = (phi_blk - phi) / (wd * P['w_infl'])
            if tb <= need:
                mem['latch'] = True; up = False; mode = 'down'
            else:
                up = True; mode = 'up'
        info.update(s=phi, s_end=phi_blk - wd * (P['T0'] + max(0.0, psi - P['psi_clear']) / P['v_ret']))
        if not up:
            rB = P['rB_ret']
    else:
        mem['latch'] = False; mem.pop('phi_prev', None)
    if wc > P['w_fast']:
        pass
    elif mem['stall'] > P['stall_ticks']:
        k = ((mem['stall'] - P['stall_ticks'] - 1) // P['wig_period']) % 2   # starts with an up-stroke
        up = (k == 0); mode = 'wiggle_up' if up else 'wiggle_down'
        rB = P['rB_slow']; top = P['psi_top_wig']
    elif w > P['w_fast']:        # spinning the wrong way fast: head-on strike reverses it
        up = P['ccw_lo'] <= p <= P['ccw_hi']
        mode = 'slow_up' if up else 'slow_down'
    else:
        up = P['slow_lo'] <= p <= P['slow_hi']
        mode = 'slow_up' if up else 'slow_down'; top = P['psi_top_slow']
    q2c = q2_for_rB(rB)
    u2 = P['kp2'] * (q2c - q[1]) - P['kd2'] * dq[1]
    if up:
        u1 = 1.0 if psi < top else P['kp1'] * (q1_for_psi(top * D2R, q[1]) - q[0]) - P['kd1'] * dq[0]
    else:
        u1 = -1.0 if psi > P['psi_bot'] + 4.0 else P['kp1'] * (q1_for_psi(P['psi_bot'] * D2R, q[1]) - q[0]) - P['kd1'] * dq[0]
    info['mode'] = mode
    return np.clip([u1, u2], -1, 1), info


def policy(observation, step, memory):
    a, _ = act(observation, memory, DEFAULT)
    return [float(a[0]), float(a[1])]
