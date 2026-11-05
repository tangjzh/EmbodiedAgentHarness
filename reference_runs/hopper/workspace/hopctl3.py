"""hopctl3: capture-point ankle balance with physical CoP limits + ankle (toe-push) hops.
Modes: stand (balance; xc_d may be ramped = lean), push, flight, land.
Static foot model (verified EXP01): u3 = (0.065*m_f*g - Fz*x_cop)/200, x_cop rel. ankle, support ~[-0.12, +0.24].
Signs: u1>0 pitches torso fwd (stance); u2>0 extends knee (q2->0 limit); u3<0 = toe-down (CoP->toe, push-off).
"""
import numpy as np
from hopper_lib import estimate, MTOT, G

MF = 5.31557

P3 = dict(
    kp_h=3.0, kd_h=0.25, p_nom=0.0, hip_margin=0.05, p_min=-0.12,  # stance torso servo; keep q1 <= -margin
    kp_hf=3.0, kd_hf=0.25, p_fl=0.0,                                 # flight torso servo
    kp_k=3.0, kd_k=0.2, q2_ref=0.03,                                 # knee: bias into straight limit
    xc_d=0.055, k_xi=1.0, cop_lo=-0.10, cop_hi=0.22,                 # CP balance
    push=0.5, push_max=25, lift_thr=0.006,                           # ankle push-off
    a3_td=0.0, kp_a=1.5, kd_a=0.08, q2_fl=-0.03,                     # flight: foot flat, knee ~straight
    contact_thr=0.003, land_ticks=8, kp_k_land=2.0,
)


def clip(u, lo=-1.0, hi=1.0):
    return float(np.clip(u, lo, hi))


def ankle_cp(st, xc_d, P):
    w0 = np.sqrt(G / max(st['zc'], 0.3))
    xi = st['xc'] + st['vxc'] / w0
    x_cop = float(np.clip(xi + P['k_xi'] * (xi - xc_d), P['cop_lo'], P['cop_hi']))
    u3 = (0.065 * MF * G - MTOT * G * x_cop) / 200.0
    return u3, x_cop, xi


def control3(obs, mem, P):
    st = estimate(obs)
    mode = mem.setdefault('mode', 'stand')
    mem['t_mode'] = mem.get('t_mode', 0) + 1
    fh = st['foot_h']
    mem['n_air'] = mem.get('n_air', 0) + 1 if fh > P['lift_thr'] else 0

    def goto(m):
        mem['mode'] = m
        mem['t_mode'] = 0

    # stance torso target keeps the hip off its q1<=0 limit
    p_st = float(np.clip(min(mem.get('p_nom', P['p_nom']), st['a1'] - P['hip_margin']), P['p_min'], 0.15))
    u1 = P['kp_h'] * (p_st - st['p']) - P['kd_h'] * st['dp']
    u2 = P['kp_k'] * (P['q2_ref'] - st['q2']) - P['kd_k'] * st['dq2']
    xc_d = mem.get('xc_d', P['xc_d'])
    u3, x_cop, xi = ankle_cp(st, xc_d, P)
    st['x_cop'], st['xi'], st['p_st'] = x_cop, xi, p_st

    if mode == 'stand':
        pass
    elif mode == 'push':
        u3 = -mem.get('push', P['push'])
        if mem['n_air'] >= 2:
            goto('flight')
        elif mem['t_mode'] > P['push_max']:
            goto('land')
    elif mode == 'flight':
        pf = mem.get('p_fl', P['p_fl'])
        u1 = P['kp_hf'] * (pf - st['p']) - P['kd_hf'] * st['dp']
        u2 = P['kp_k'] * (P['q2_fl'] - st['q2']) - P['kd_k'] * st['dq2']
        q3d = float(np.clip(st['a2'] - mem.get('a3_td', P['a3_td']), -0.78, 0.78))
        u3 = P['kp_a'] * (q3d - st['q3']) - P['kd_a'] * st['dq3']
        if fh < P['contact_thr'] and mem['t_mode'] > 2:
            goto('land')
    elif mode == 'land':
        u2 = P['kp_k_land'] * (P['q2_ref'] - st['q2']) - P['kd_k'] * st['dq2']
        if mem['t_mode'] >= P['land_ticks']:
            goto('stand')
    return np.array([clip(u1), clip(u2), clip(u3)]), st
