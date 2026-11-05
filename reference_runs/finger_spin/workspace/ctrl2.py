"""Face-tracking ratchet controller (v2). Uses only the observation.
psi = fingertip-centre angle about proximal joint B on arc radius rB (q2 held).
phase p = th mod 180deg (decreasing when spinning CW = rewarded).
PSI_LOW[p]: lowest psi on the arc that overlaps the bar (the trailing face of the left arm for CW rotation).
PUSH (p_pred in [p_end(w), p_start]): drive fingertip up to psi_low(p_pred)+delta (full effort until there).
Otherwise return to psi_wait below the sweep disc.  Slow-bar spin-up: CCW nudge for '/' orientation."""
import os, numpy as np

B = np.array([-0.2, 0.4]); H = np.array([0.2, 0.4]); L1 = 0.18; S = 0.1455
S_A, S_B, R_TIP, R_CAP = 0.13, 0.161, 0.03, 0.04
D2R = np.pi / 180
WS = os.path.dirname(os.path.abspath(__file__))

DEFAULT = dict(rB=0.29, psi_wait=-27.0, p_start=110.0, p_face=98.0, psi_clamp=-24.5, p_end_min=10.0,
               k_ret=4.3, ret_margin=15.0,
               delta=4.0, T_lead=0.03, kp1=4.0, kd1=0.03, kp2=4.0, kd2=0.05, kp_push=12.0,
               w_slow=6.0, w_nudge=1.0, nudge_lo=112.0, nudge_hi=176.0, psi_nudge=-14.0)


def q2_for_rB(rB):
    c = (rB ** 2 - L1 ** 2 - S ** 2) / (2 * L1 * S)
    return float(np.arccos(np.clip(c, -1, 1)))


def fk(q, s=S):
    a2 = q[0] + q[1]
    D = B + L1 * np.array([np.cos(q[0]), np.sin(q[0])])
    return D + s * np.array([np.cos(a2), np.sin(a2)]), D


def psi_of(q):
    d = fk(q)[0] - B
    return float(np.arctan2(d[1], d[0]))


def q1_for_psi(psi, q2):
    return psi - np.arctan2(S * np.sin(q2), L1 + S * np.cos(q2))


def _segdist(p1, q1, p2, q2):
    d1 = q1 - p1; d2 = q2 - p2; r = p1 - p2
    a = d1 @ d1; e = d2 @ d2; f = d2 @ r; c = d1 @ r; b = d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
    t = (b * s + f) / e
    if t < 0:
        t = 0.0; s = np.clip(-c / a, 0, 1)
    elif t > 1:
        t = 1.0; s = np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def gap(q, th):
    _, D = fk(q)
    a2 = q[0] + q[1]; u2 = np.array([np.cos(a2), np.sin(a2)])
    pa, pb = D + S_A * u2, D + S_B * u2
    u = np.array([-np.sin(th), np.cos(th)]); v = np.array([np.cos(th), np.sin(th)])
    g = 1e9
    for sx in (0.02, -0.02):
        c = H + sx * v
        g = min(g, _segdist(pa, pb, c - 0.09 * u, c + 0.09 * u))
    return g - R_TIP - R_CAP


def build_table(rB):
    fn = os.path.join(WS, f'psilow_{rB:.3f}.npy')
    if os.path.exists(fn):
        return np.load(fn)
    q2 = q2_for_rB(rB)
    psis = np.arange(-40, 30.01, 0.25)
    tab = np.full(180, np.nan)
    for i in range(180):
        th = i * D2R
        for ps in psis:
            if gap([q1_for_psi(ps * D2R, q2), q2], th) < 0:
                tab[i] = ps; break
    np.save(fn, tab)
    return tab


_TABLES = {}


def psi_low(p_deg, rB):
    if rB not in _TABLES:
        _TABLES[rB] = build_table(rB)
    tab = _TABLES[rB]
    i0 = int(np.floor(p_deg)) % 180; i1 = (i0 + 1) % 180; f = p_deg - np.floor(p_deg)
    a, b = tab[i0], tab[i1]
    if np.isnan(a) or np.isnan(b):
        return a if not np.isnan(a) else b
    return a * (1 - f) + b * f


def decode(obs):
    pos = np.asarray(obs['position'], float); vel = np.asarray(obs['velocity'], float)
    return pos[:2], vel[:2], float(np.arctan2(-pos[2], pos[3])), float(vel[2])


def act(obs, mem, P):
    q, dq, th, w = decode(obs)
    rB = P['rB']
    q2c = q2_for_rB(rB)
    u2 = P['kp2'] * (q2c - q[1]) - P['kd2'] * dq[1]
    psi = psi_of(q) / D2R
    p = (th / D2R) % 180.0
    pp = ((th + w * P['T_lead']) / D2R) % 180.0
    aw = abs(w)
    # end of push window: must leave time to return below the next arm (arrives ~p=165)
    p_end = min(95.0, max(P['p_end_min'], aw * P['k_ret'] - P['ret_margin']))
    mode = 'wait'; u1 = None
    if w < P['w_slow'] and p_end <= pp <= P['p_start']:
        pl = psi_low(pp, rB)
        if not np.isnan(pl):
            tgt = pl + P['delta'] if pp <= P['p_face'] else min(pl - 1.5, P['psi_clamp'])
            mode = 'push'
            pl2 = psi_low((pp - 2.0) % 180.0, rB)
            slope = 0.0 if np.isnan(pl2) else (pl2 - pl) / -2.0          # dpsi/dp (deg/deg)
            vff = slope * w                                                # rad/s of psi
            u1 = 1.0 if psi < tgt - 2.0 else vff / 12.0 + P['kp_push'] * (tgt - psi) * D2R - P['kd1'] * (dq[0] - vff)
    elif w >= P['w_slow'] and p_end <= pp <= P['p_start']:
        # bar spinning CCW into the window: head-on push to reverse it
        pl = psi_low(pp, rB)
        if not np.isnan(pl):
            mode = 'headon'; u1 = 1.0 if psi < pl + P['delta'] else 0.0
    elif aw < P['w_nudge'] and P['nudge_lo'] < p < P['nudge_hi']:
        mode = 'nudge'
        u1 = 1.0 if psi < P['psi_nudge'] else None
    if u1 is None:
        q1w = q1_for_psi(P['psi_wait'] * D2R, q[1])
        e = q1w - q[0]
        u1 = -1.0 if e < -0.12 else P['kp1'] * e - P['kd1'] * dq[0]
        if mode != 'nudge':
            mode = 'wait'
    return np.clip([u1, u2], -1, 1), dict(psi=psi, p=p, pp=pp, mode=mode, p_end=p_end)


MODES = {'wait': 0, 'push': 1, 'headon': 2, 'nudge': 3}
