"""EXP01 (episode 1): system ID. stand -> pitch steps -> squat -> 3 single hops (increasing thrust) -> stand."""
import numpy as np
import robot_api as robot
from hopctl import control, DEFAULT_P
from runlib import parse, save_log

P = dict(DEFAULT_P)
mem = {'mode': 'stand', 'q2_ref': 0.0, 'p_d': 0.0}
obs = np.asarray(robot.observe(), float)
log = {k: [] for k in ['obs', 'act', 'rew', 'mode', 'foot_h', 'heel_h', 'toe_h', 'xc', 'zc', 'vxc', 'vzc', 'a3']}
hops = [(0.5, 0.1), (0.8, 0.25), (1.0, 0.4)]
hop_i = 0
t_stand = 0
ret = 0.0
modes_seq = []
for t in range(1000):
    # --- experiment program ---
    if t < 100:
        mem['p_d'] = 0.0
    elif t < 140:
        mem['p_d'] = 0.05
    elif t < 180:
        mem['p_d'] = -0.05
    else:
        mem['p_d'] = 0.0
    if 220 <= t < 320:
        mem['q2_ref'] = -0.2 * (1 - np.cos(2 * np.pi * (t - 220) / 100.0))
    if mem['mode'] == 'stand':
        t_stand += 1
        if t >= 320:
            mem['q2_ref'] = min(0.0, mem.get('q2_ref', 0.0) + 0.005)
        if t >= 330 and t_stand > 80 and hop_i < len(hops):
            mem['u2_thrust'], mem['u3_push'] = hops[hop_i]
            print(f't={t} start hop {hop_i} thrust={hops[hop_i]}')
            hop_i += 1
            mem['mode'] = 'crouch'; mem['t_mode'] = 0
            t_stand = 0
    else:
        t_stand = 0
    a, st = control(obs, mem, P)
    r = robot.step(a.tolist())
    obs2, rew, term, trunc, raw = parse(r)
    ret += rew
    for k, v in [('obs', obs), ('act', a), ('rew', rew), ('mode', mem['mode']), ('foot_h', st['foot_h']),
                 ('heel_h', st['heel_h']), ('toe_h', st['toe_h']), ('xc', st['xc']), ('zc', st['zc']),
                 ('vxc', st['vxc']), ('vzc', st['vzc']), ('a3', st['a3'])]:
        log[k].append(v)
    if not modes_seq or modes_seq[-1][1] != mem['mode']:
        modes_seq.append((t, mem['mode']))
    obs = obs2
    if term or trunc:
        print('TERMINAL at t', t, 'term', term, 'trunc', trunc)
        break
save_log('exp01', log)
O = np.array(log['obs'])
print('steps', len(O), 'return', round(ret, 2))
print('mode transitions:', modes_seq[:60])
for t0 in [0, 5, 10, 20, 50, 99, 120, 139, 160, 179, 219, 270, 319]:
    if t0 < len(O):
        print(t0, 'z %.3f p %.3f q %.3f %.3f %.3f vx %.3f footh %.4f heel %.4f toe %.4f xc %.3f act %s' % (
            O[t0, 0], O[t0, 1], O[t0, 2], O[t0, 3], O[t0, 4], O[t0, 5], log['foot_h'][t0], log['heel_h'][t0],
            log['toe_h'][t0], log['xc'][t0], np.round(log['act'][t0], 3)))
print('pitch range', O[:, 1].min(), O[:, 1].max(), ' z range', O[:, 0].min(), O[:, 0].max())
print('raw last:', {k: v for k, v in raw.items() if k not in ('observation',)} if isinstance(raw, dict) else raw)
