"""EXP04 (episode 4): verify CP stand (150 ticks), then single lean+toe-push hops (conservative -> aggressive)
with recovery stands. Logs saved incrementally to logs/exp04.npz."""
import numpy as np
import robot_api as robot
from hopctl4 import control4, P4
from runlib import parse, save_log

obs = np.asarray(robot.observe()['observation'] if isinstance(robot.observe(), dict) else robot.observe(), float)
P = dict(P4)
mem = {'mode': 'stand'}
trials = [(0.10, 0.30), (0.10, 0.45), (0.13, 0.45), (0.13, 0.60), (0.16, 0.60), (0.16, 0.80), (0.19, 0.80), (0.19, 1.0)]
K = ['obs', 'act', 'mode', 'fh', 'xc', 'vxc', 'a1', 'a2', 'a3', 'cop', 'xi', 'phase', 'rew', 'zc']
L = {k: [] for k in K}
prog = 'settle'; t_prog = 0; ti = -1; settle_n = 150
ret = 0.0; x = 0.0; events = []
raw = {}
for t in range(1000):
    t_prog += 1
    if prog == 'settle':
        mem['xc_d'] = P['xc_stand']
        if t_prog >= settle_n and mem['mode'] == 'stand':
            ti += 1
            if ti < len(trials):
                prog = 'lean'; t_prog = 0
                events.append(dict(trial=ti, lean=trials[ti][0], push=trials[ti][1], t_lean=t, x0=x))
    elif prog == 'lean':
        k = min(1.0, t_prog / 25.0)
        mem['xc_d'] = P['xc_stand'] + k * (trials[ti][0] - P['xc_stand'])
        if t_prog >= 35:
            mem['push'] = trials[ti][1]; mem['push_now'] = True
            prog = 'hop'; t_prog = 0
            events[-1]['t_push'] = t
    elif prog == 'hop':
        if mem['mode'] == 'flight' and 't_lift' not in events[-1]:
            events[-1]['t_lift'] = t
        if mem['mode'] == 'land' and 't_land' not in events[-1]:
            events[-1]['t_land'] = t
        if mem['mode'] == 'stand' and t_prog > 3:
            mem['xc_d'] = P['xc_stand']
            prog = 'settle'; t_prog = 0; settle_n = 80
            events[-1]['t_stand'] = t
    a, st = control4(obs, mem, P)
    obs2, rew, term, trunc, raw = parse(robot.step(a.tolist()))
    ret += rew; x += obs[5] * 0.008
    for k, v in [('obs', obs), ('act', a), ('mode', mem['mode']), ('fh', st['foot_h']), ('xc', st['xc']), ('vxc', st['vxc']),
                 ('a1', st['a1']), ('a2', st['a2']), ('a3', st['a3']), ('cop', st['x_cop']), ('xi', st['xi']),
                 ('phase', prog), ('rew', rew), ('zc', st['zc'])]:
        L[k].append(v)
    obs = obs2
    if t % 100 == 99:
        save_log('exp04', L)
    if term or trunc:
        print('TERMINAL at t', t, 'prog', prog, 'mode', mem['mode'])
        break
save_log('exp04', L)
O = np.array(L['obs']); A = np.array(L['act'])
print('steps', len(O), 'return %.2f' % ret, 'x_est %.3f' % x, 'final', {k: v for k, v in raw.items() if k not in ('observation',)})
for t0 in [0, 20, 50, 100, 149]:
    if t0 < len(O):
        o = O[t0]
        print('  stand %4d z %.3f p %+.3f dp %+.3f q %+.3f %+.3f %+.3f vx %+.3f xc %+.3f cop %+.3f u %s' % (
            t0, o[0], o[1], o[7], o[2], o[3], o[4], o[5], L['xc'][t0], L['cop'][t0], np.round(A[t0], 3)))
for e in events:
    print('TRIAL', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()})
    tl = e.get('t_lift'); td = e.get('t_land'); tp = e.get('t_push')
    for nm, tt in [('push', tp), ('liftoff', tl), ('land', td)]:
        if tt is not None and tt < len(O):
            o = O[tt]
            print('   %-8s t%4d z %.3f vx %+.2f vz %+.2f p %+.3f dp %+.2f xc %+.3f vxc %+.2f a1 %+.3f a2 %+.3f q1 %+.2f q3 %+.2f' % (
                nm, tt, o[0], o[5], o[6], o[1], o[7], L['xc'][tt], L['vxc'][tt], L['a1'][tt], L['a2'][tt], o[2], o[4]))
    s0 = e['t_lean']; s1 = min(len(O), e.get('t_stand', len(O)) + 60)
    print('   p range [%+.3f, %+.3f]  x gain %.3f  max fh %.3f  max z %.3f' % (
        O[s0:s1, 1].min(), O[s0:s1, 1].max(), O[s0:s1, 5].sum() * 0.008, max(L['fh'][s0:s1]), O[s0:s1, 0].max()))
# dense trace of the last trial (the most informative if it failed)
if events:
    e = events[-1]
    t0s = e.get('t_push', e['t_lean'])
    for t0 in range(max(0, t0s - 4), len(O), 3):
        o = O[t0]
        print('  %4d %-6s z %.3f p %+.3f dp %+.2f q %+.2f %+.2f %+.2f vx %+.2f vz %+.2f xc %+.3f xi %+.3f cop %+.3f fh %.3f a1 %+.2f u %s' % (
            t0, L['mode'][t0], o[0], o[1], o[7], o[2], o[3], o[4], o[5], o[6], L['xc'][t0], L['xi'][t0], L['cop'][t0], L['fh'][t0], L['a1'][t0], np.round(A[t0], 2)))
        if t0 > t0s + 90:
            break
