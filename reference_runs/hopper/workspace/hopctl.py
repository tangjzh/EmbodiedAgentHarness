"""Phase-based hopper controller core (v2: posture-consistent stance).
Modes: stand, crouch, thrust, flight, land.
Verified signs (EXP01): u_hip>0 pitches torso forward (only possible until q_thigh hits 0 limit),
p = a_thigh + q_thigh, q_thigh<=0 => torso can only lean BACK relative to thigh.
Stance posture law: ankle servos absolute shank angle a2 -> a2_d = a1_d - q2 with a1_d = p_d + hip_flex,
so thigh stays tilted forward of torso and the hip keeps authority.
"""
import numpy as np
from hopper_lib import estimate, MTOT, G

DEFAULT_P = dict(
    kp_h=3.0, kd_h=0.25, p_d=0.0, hip_flex=0.1,   # torso pitch servo via hip; thigh forward margin
    kp_k=3.0, kd_k=0.2, q2_stand=0.0,             # knee PD in posture modes
    kp_s=3.0, kd_s=0.4, u3_lo=-0.25, u3_hi=0.15,  # ankle posture servo in stance (a2), torque clip (CoP limits)
    q2_crouch=-0.5, crouch_ticks=40,              # crouch
    u2_thrust=0.6, thrust_ankle='posture', u3_push=0.2, thrust_max=40,
    q2_td=-0.15, kp_kf=3.0, kd_kf=0.2,            # flight knee
    a3_td=0.0, kp_a=1.5, kd_a=0.08,               # flight ankle (absolute foot angle target, + = toe down)
    q2_land=-0.35, kp_land=2.5, kd_land=0.2,      # landing knee spring
    contact_thr=0.008,
    continuous=False,
)


def clip(u, lo=-1.0, hi=1.0):
    return float(np.clip(u, lo, hi))


def control(obs, mem, P):
    st = estimate(obs)
    mode = mem.setdefault('mode', 'stand')
    mem['t_mode'] = mem.get('t_mode', 0) + 1
    contact = st['foot_h'] < P['contact_thr']
    mem['nc'] = 0 if contact else mem.get('nc', 0) + 1
    mem['ncon'] = mem.get('ncon', 0) + 1 if contact else 0
    p_d = mem.get('p_d', P['p_d'])
    a1_d = p_d + P['hip_flex']
    u1 = P['kp_h'] * (p_d - st['p']) - P['kd_h'] * st['dp']
    u2 = 0.0
    u3 = 0.0

    def goto(m):
        mem['mode'] = m
        mem['t_mode'] = 0

    def ankle_posture(a2_d):
        return clip(P['kp_s'] * (a2_d - st['a2']) - P['kd_s'] * st['da2'], P['u3_lo'], P['u3_hi'])

    if mode in ('stand', 'crouch'):
        if mode == 'crouch':
            k = min(1.0, mem['t_mode'] / float(P['crouch_ticks']))
            q2r = mem.get('q2_start', 0.0) + k * (P['q2_crouch'] - mem.get('q2_start', 0.0))
            mem['q2_ref'] = q2r
            if k >= 1.0 and st['q2'] <= P['q2_crouch'] + 0.05:
                mem['a2_hold'] = st['a2']
                goto('thrust')
        else:
            q2r = mem.get('q2_ref', P['q2_stand'])
        u2 = P['kp_k'] * (q2r - st['q2']) - P['kd_k'] * st['dq2']
        u3 = ankle_posture(a1_d - q2r)
    elif mode == 'thrust':
        u2 = mem.get('u2_thrust', P['u2_thrust'])
        am = mem.get('thrust_ankle', P['thrust_ankle'])
        if am == 'posture':
            u3 = ankle_posture(a1_d - st['q2'])
        elif am == 'hold':
            u3 = ankle_posture(mem.get('a2_hold', st['a2']))
        else:
            u3 = -mem.get('u3_push', P['u3_push'])
        if mem['nc'] >= 2:
            goto('flight')
        elif mem['t_mode'] > P['thrust_max']:
            mem['q2_ref'] = st['q2']
            goto('stand')
    elif mode == 'flight':
        u2 = P['kp_kf'] * (P['q2_td'] - st['q2']) - P['kd_kf'] * st['dq2']
        q3d = np.clip(st['a2'] - mem.get('a3_td', P['a3_td']), -0.75, 0.75)
        u3 = clip(P['kp_a'] * (q3d - st['q3']) - P['kd_a'] * st['dq3'])
        if contact:
            goto('land')
    elif mode == 'land':
        u2 = P['kp_land'] * (P['q2_land'] - st['q2']) - P['kd_land'] * st['dq2']
        u3 = ankle_posture(a1_d - st['q2'])
        if mem['t_mode'] > 3 and st['vzc'] >= 0.0:
            if mem.get('continuous', P['continuous']):
                mem['a2_hold'] = st['a2']
                goto('thrust')
            else:
                mem['q2_ref'] = st['q2']
                goto('stand')
    return np.array([clip(u1), clip(u2), clip(u3)]), st
