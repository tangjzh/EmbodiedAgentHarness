"""Phase-window ratchet controller for finger/spin (uses only the observation).
Fingertip rides an arc of radius rB about the proximal joint B; angle psi about B.
Bar phase p = th mod pi (decreasing when spinning clockwise = rewarded direction).
CW push: when predicted phase in [p_off, p_on] drive u1=+1 (fingertip up under the left arm's trailing face),
otherwise servo back to psi_wait below the sweep disc.  Low-speed spin-up handled by the same logic plus
head-on reversal strike when the bar turns CCW."""
import numpy as np

B = np.array([-0.2, 0.4]); L1 = 0.18; S = 0.1455
D2R = np.pi / 180

DEFAULT = dict(rB=0.29, psi_wait=-27.0, psi_max=0.0, p_on=108.0, p_off=40.0, T_lead=0.02,
               kp1=3.0, kd1=0.04, kp2=4.0, kd2=0.05, ret_bang=-8.0,
               # CCW reversal strike: launch when predicted phase (CCW, increasing) reaches p_strike
               p_strike=12.0, T_flight=0.06, w_lo=3.0)


def q2_for_rB(rB):
    c = (rB ** 2 - L1 ** 2 - S ** 2) / (2 * L1 * S)
    return float(np.arccos(np.clip(c, -1, 1)))


def tip(q):
    a2 = q[0] + q[1]
    return B + L1 * np.array([np.cos(q[0]), np.sin(q[0])]) + S * np.array([np.cos(a2), np.sin(a2)])


def psi_of(q):
    d = tip(q) - B
    return float(np.arctan2(d[1], d[0]))


def q1_for_psi(psi, q2):
    # fingertip angle about B = q1 + atan2(S sin q2, L1 + S cos q2)
    return psi - np.arctan2(S * np.sin(q2), L1 + S * np.cos(q2))


def decode(obs):
    pos = np.asarray(obs['position'], float); vel = np.asarray(obs['velocity'], float)
    q = pos[:2]; dq = vel[:2]
    th = float(np.arctan2(-pos[2], pos[3])); w = float(vel[2])
    return q, dq, th, w


def act(obs, mem, P=None):
    P = P or mem.get('P') or DEFAULT
    q, dq, th, w = decode(obs)
    q2c = q2_for_rB(P['rB'])
    u2 = P['kp2'] * (q2c - q[1]) - P['kd2'] * dq[1]
    psi = psi_of(q)
    p = (th % np.pi) / D2R                      # degrees in [0,180)
    pp = ((th + w * P['T_lead']) % np.pi) / D2R  # predicted phase
    mode = 'wait'
    u1 = None
    if w <= P['w_lo']:
        # clockwise or slow: CW push window (handles wrap since p_off<p_on within [0,180))
        if P['p_off'] <= pp <= P['p_on'] and psi < P['psi_max'] * D2R:
            u1 = 1.0; mode = 'push'
        elif w > -P['w_lo'] and 110.0 < p < 178.0:
            # slow and arm in lower-left "/" orientation: nudge it CCW by pushing the end up
            if psi < -12 * D2R:
                u1 = 1.0; mode = 'nudge'
    else:
        # bar turning CCW: head-on reversal strike timed on predicted phase
        pf = ((th + w * P['T_flight']) % np.pi) / D2R
        if (P['p_strike'] <= pf <= P['p_strike'] + 60.0) and psi < P['psi_max'] * D2R:
            u1 = 1.0; mode = 'strike'
    if u1 is None:
        q1w = q1_for_psi(P['psi_wait'] * D2R, q[1])
        e = q1w - q[0]
        u1 = P['kp1'] * e - P['kd1'] * dq[0]
        if e < -0.15:
            u1 = -1.0
    mem['mode'] = mode
    return np.clip([u1, u2], -1, 1), dict(psi=psi, p=p, pp=pp, mode=mode)
