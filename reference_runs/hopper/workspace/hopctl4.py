"""hopctl4: capture-point (CP) ankle balance + lean / toe-push hop state machine.
Signs (EXP01/02): u1>0 pitches torso fwd in stance (hip limit q1<=0 => p<=a1); u2>0 extends knee (limit q2<=0);
u3<0 = toe-down torque (CoP -> toe, push-off). Static foot model: u3 = (0.065*MF*g - Fz*x_cop)/200.
Straight leg + upright torso: xc ~= 0.022 + 0.45*sin(leg tilt).  COM height ~0.56 m -> w0 ~ 4.2 rad/s.
Modes: stand (CP balance to mem['xc_d']), push (ankle toe-down), flight (foot flat, torso servo), land.
If P['auto']: stand -> (ramp xc_d to xc_lean) -> push when xc>=push_xc  => repeated forward hops.
"""
import numpy as np
from hopper_lib import estimate, MTOT, G

MF = 5.31557

P4 = dict(
    kp_h=3.0, kd_h=0.25, p_nom=0.0, hip_margin=0.05, p_min=-0.12, p_max=0.15,  # stance torso servo
    kp_hf=3.0, kd_hf=0.25, p_fl=0.0,                                            # flight torso servo
    kp_k=3.0, kd_k=0.2, q2_ref=0.03,                                            # knee into straight limit
    xc_stand=0.06, k_xi=1.0, cop_lo=-0.10, cop_hi=0.22,                         # CP balance
    push=0.5, push_max=20, lift_thr=0.006,                                      # toe push
    a3_td=0.0, kp_a=1.5, kd_a=0.08, q2_fl=0.03,                                 # flight
    contact_thr=0.003, land_ticks=6,
    xc_gait=0.22, xi_push=0.20, min_stance=5, k_push=0.0, push_lo=0.3,          # continuous gait (mem['gait_on'])
    xc_ref=0.12, kN=0.0, N_lo=8, N_hi=20, xc_hard=0.16, kV=0.0, vxc_ref=0.45,   # adaptive push (duration/magnitude)
    lift_end=True,                                                              # end push on detected liftoff
)


def clip(u, lo=-1.0, hi=1.0):
    return float(np.clip(u, lo, hi))


def ankle_cp(st, xc_d, P):
    w0 = np.sqrt(G / max(st['zc'], 0.3))
    xi = st['xc'] + st['vxc'] / w0
    x_cop = float(np.clip(xi + P['k_xi'] * (xi - xc_d), P['cop_lo'], P['cop_hi']))
    u3 = (0.065 * MF * G - MTOT * G * x_cop) / 200.0
    return u3, x_cop, xi


def goto(mem, m):
    mem['mode'] = m
    mem['t_mode'] = 0
    mem.setdefault('events', []).append((mem.get('t', 0), m))


def control4(obs, mem, P):
    st = estimate(obs)
    mem['t'] = mem.get('t', -1) + 1
    mode = mem.setdefault('mode', 'stand')
    mem['t_mode'] = mem.get('t_mode', 0) + 1
    fh = st['foot_h']
    mem['n_air'] = mem.get('n_air', 0) + 1 if fh > P['lift_thr'] else 0

    if mem.get('gait_on') and mode in ('stand', 'land'):
        mem['xc_d'] = P['xc_gait']
    mem['t_stance'] = mem.get('t_stance', 0) + 1 if mode in ('stand', 'land') else 0

    p_st = float(np.clip(min(mem.get('p_nom', P['p_nom']), st['a1'] - P['hip_margin']), P['p_min'], P['p_max']))
    u1 = P['kp_h'] * (p_st - st['p']) - P['kd_h'] * st['dp']
    u2 = P['kp_k'] * (P['q2_ref'] - st['q2']) - P['kd_k'] * st['dq2']
    xc_d = mem.get('xc_d', P['xc_stand'])
    u3, x_cop, xi = ankle_cp(st, xc_d, P)
    st['x_cop'], st['xi'], st['p_st'] = x_cop, xi, p_st

    if mode == 'stand':
        if mem.get('gait_on') and mem['t_stance'] >= P['min_stance'] and xi >= P['xi_push']:
            ex = st['xc'] - P['xc_ref']
            ev = st['vxc'] - P['vxc_ref']
            mem['push_n'] = int(np.clip(round(P['push_max'] + P['kN'] * ex / 0.01 + P['kV'] * ev / 0.1), P['N_lo'], P['N_hi']))
            mem['push'] = float(np.clip(P['push'] + P['k_push'] * max(0.0, st['xc'] - P['xc_hard']), P['push_lo'], 1.0))
            mem['push_now'] = True
        if mem.pop('push_now', False):
            mem.setdefault('hops', []).append(dict(t=mem['t'], xc=st['xc'], xi=xi, vxc=st['vxc'], a2=st['a2'],
                                                   p=st['p'], push=mem.get('push', P['push'])))
            goto(mem, 'push')
            mode = 'push'
    if mode == 'push':
        u3 = -mem.get('push', P['push'])
        if P['lift_end'] and mem['n_air'] >= 2:
            goto(mem, 'flight')
        elif mem['t_mode'] > mem.get('push_n', P['push_max']):
            goto(mem, 'land')
    elif mode == 'flight':
        pf = mem.get('p_fl', P['p_fl'])
        u1 = P['kp_hf'] * (pf - st['p']) - P['kd_hf'] * st['dp']
        u2 = P['kp_k'] * (P['q2_fl'] - st['q2']) - P['kd_k'] * st['dq2']
        q3d = float(np.clip(st['a2'] - mem.get('a3_td', P['a3_td']), -0.78, 0.78))
        u3 = P['kp_a'] * (q3d - st['q3']) - P['kd_a'] * st['dq3']
        if fh < P['contact_thr'] and mem['t_mode'] > 2:
            goto(mem, 'land')
    elif mode == 'land':
        if mem['t_mode'] >= P['land_ticks']:
            goto(mem, 'stand')
    return np.array([clip(u1), clip(u2), clip(u3)]), st
