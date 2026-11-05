"""EXP03 (episode 3): hopctl3 CP balance + single ankle-push hops (increasing lean/push) with recovery stands."""
import numpy as np
import robot_api as robot
from hopctl3 import control3, P3
from runlib import parse, save_log

r0 = robot.reset()
print('reset ->', str(r0)[:160])
obs = np.asarray(robot.observe(), float)
P = dict(P3)
mem = {'mode': 'stand'}
trials = [(0.10, 0.35), (0.12, 0.5), (0.15, 0.5), (0.15, 0.7), (0.18, 0.7), (0.18, 0.9), (0.20, 0.9), (0.20, 1.0)]
L = {k: [] for k in ['obs', 'act', 'mode', 'fh', 'xc', 'vxc', 'a1', 'a2', 'a3', 'cop', 'xi', 'phase']}
prog = 'settle'; t_prog = 0; ti = -1; settle_n = 60
ret = 0.0; x = 0.0; events = []
for t in range(1000):
    t_prog += 1
    if prog == 'settle':
        mem['xc_d'] = P['xc_d']
        if t_prog >= settle_n and mem['mode'] == 'stand':
            ti += 1
            if ti < len(trials):
                prog = 'lean'; t_prog = 0
                events.append(dict(trial=ti, lean=trials[ti][0], push=trials[ti][1], t_lean=t, x0=x))
    elif prog == 'lean':
        k = min(1.0, t_prog / 25.0)
        mem['xc_d'] = P['xc_d'] + k * (trials[ti][0] - P['xc_d'])
        if t_prog >= 35:
            mem['push'] = trials[ti][1]; mem['mode'] = 'push'; mem['t_mode'] = 0
            prog = 'hop'; t_prog = 0
            events[-1]['t_push'] = t
    elif prog == 'hop':
        if mem['mode'] == 'flight' and 't_lift' not in events[-1]:
            events[-1]['t_lift'] = t
        if mem['mode'] == 'land' and 't_land' not in events[-1]:
            events[-1]['t_land'] = t
        if mem['mode'] == 'stand':
            mem['xc_d'] = P['xc_d']
            prog = 'settle'; t_prog = 0; settle_n = 90
            events[-1]['t_stand'] = t
    a, st = control3(obs, mem, P)
    obs2, rew, term, trunc, raw = parse(robot.step(a.tolist()))
    ret += rew; x += obs[5] * 0.008
    for k, v in [('obs', obs), ('act', a), ('mode', mem['mode']), ('fh', st['foot_h']), ('xc', st['xc']), ('vxc', st['vxc']),
                 ('a1', st['a1']), ('a2', st['a2']), ('a3', st['a3']), ('cop', st['x_cop']), ('xi', st['xi']), ('phase', prog)]:
        L[k].append(v)
    obs = obs2
    if term or trunc:
        print('TERMINAL at t', t, 'prog', prog, 'mode', mem['mode'])
        break
save_log('exp03', L)
O = np.array(L['obs']); A = np.array(L['act'])
print('steps', len(O), 'return %.2f' % ret, 'x_est %.3f' % x, 'final', {k: v for k, v in raw.items() if k not in ('observation',)})
for e in events:
    print('TRIAL', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()})
    tl = e.get('t_lift'); td = e.get('t_land')
    if tl is not None:
        o = O[tl]
        print('   liftoff: vx %+.2f vz %+.2f p %+.3f dp %+.2f xc %+.3f a1 %+.3f q3 %+.2f' % (o[5], o[6], o[1], o[7], L['xc'][tl], L['a1'][tl], o[4]))
    if td is not None and td < len(O):
        o = O[td]
        print('   land:    vx %+.2f vz %+.2f p %+.3f dp %+.2f xc %+.3f a1 %+.3f q3 %+.2f' % (o[5], o[6], o[1], o[7], L['xc'][td], L['a1'][td], o[4]))
    s0 = e['t_lean']; s1 = min(len(O), e.get('t_stand', len(O)) + 60)
    print('   p range [%+.3f, %+.3f]  x gain %.3f  max fh %.3f' % (O[s0:s1, 1].min(), O[s0:s1, 1].max(), O[s0:s1, 5].sum() * 0.008, max(L['fh'][s0:s1])))
# dense trace around first two hops
for e in events[:3]:
    tp = e.get('t_push')
    if tp is None:
        continue
    for t0 in range(tp - 2, min(len(O), tp + 40), 3):
        o = O[t0]
        print('  %4d %-6s z %.3f p %+.3f dp %+.2f q %+.2f %+.2f %+.2f vx %+.2f vz %+.2f xc %+.3f cop %+.3f fh %.3f a1 %+.2f u %s' % (
            t0, L['mode'][t0], o[0], o[1], o[7], o[2], o[3], o[4], o[5], o[6], L['xc'][t0], L['cop'][t0], L['fh'][t0], L['a1'][t0], np.round(A[t0], 2)))
for t0 in [0, 10, 30, 59]:
    o = O[t0]
    print('  stand %4d z %.3f p %+.3f q %+.3f %+.3f %+.3f xc %+.3f cop %+.3f u %s' % (t0, o[0], o[1], o[2], o[3], o[4], L['xc'][t0], L['cop'][t0], np.round(A[t0], 3)))
