"""Reusable experiment runner for hopctl4-style controllers (one episode = current robot state)."""
import numpy as np
import robot_api as robot
from runlib import parse, save_log

K = ['obs', 'act', 'mode', 'fh', 'xc', 'vxc', 'a1', 'a2', 'a3', 'cop', 'xi', 'rew', 'zc']


def run(ctrl, P, name, gait_start=100, max_ticks=1000, trace_last=70, schedule=None, verbose_hops=40):
    o = robot.observe()
    obs = np.asarray(o['observation'] if isinstance(o, dict) else o, float)
    mem = {'mode': 'stand', 'xc_d': P['xc_stand']}
    L = {k: [] for k in K}
    ret = 0.0; x = 0.0; raw = {}
    hops = []; prev_mode = 'stand'
    for t in range(max_ticks):
        if t == gait_start:
            mem['gait_on'] = True
        if schedule is not None:
            schedule(t, mem, P)
        a, st = ctrl(obs, mem, P)
        m = mem['mode']
        if m != prev_mode:
            if m == 'push':
                hops.append(dict(t_push=t, xc=st['xc'], xi=st['xi'], vxc=st['vxc'], vx=obs[5], a2=st['a2'], p=obs[1], push=mem.get('push')))
            elif m == 'flight' and hops:
                hops[-1].update(t_lift=t, l_vx=obs[5], l_vz=obs[6], l_dp=obs[7], l_vxc=st['vxc'], l_a2=st['a2'], l_q3=obs[4], l_xc=st['xc'])
            elif m == 'land' and hops and prev_mode == 'flight':
                hops[-1].update(t_land=t, d_xc=st['xc'], d_xi=st['xi'], d_vxc=st['vxc'], d_p=obs[1], d_dp=obs[7], d_a2=st['a2'], d_z=obs[0])
            elif m == 'land' and hops and prev_mode == 'push':
                hops[-1].update(no_lift=t)
            prev_mode = m
        obs2, rew, term, trunc, raw = parse(robot.step(a.tolist()))
        ret += rew; x += obs[5] * 0.008
        for k, v in [('obs', obs), ('act', a), ('mode', m), ('fh', st['foot_h']), ('xc', st['xc']), ('vxc', st['vxc']),
                     ('a1', st['a1']), ('a2', st['a2']), ('a3', st['a3']), ('cop', st.get('x_cop', 0.0)), ('xi', st.get('xi', 0.0)),
                     ('rew', rew), ('zc', st['zc'])]:
            L[k].append(v)
        obs = obs2
        if t % 100 == 99:
            save_log(name, L)
        if term or trunc:
            print('TERMINAL at t', t, 'mode', m)
            break
    save_log(name, L)
    O = np.array(L['obs']); A = np.array(L['act'])
    print('%s: steps %d return %.2f x_est %.3f final %s' % (name, len(O), ret, x, {k: v for k, v in raw.items() if k not in ('observation',)}))
    print('hops:', len(hops))
    for i, h in enumerate(hops[:verbose_hops]):
        s = 'H%02d t%4d push xc%+.3f xi%+.3f vxc%+.2f a2%+.2f p%+.3f u%.2f' % (i, h['t_push'], h['xc'], h['xi'], h['vxc'], h['a2'], h['p'], h['push'] or 0)
        if 't_lift' in h:
            s += ' | lift +%d vx%+.2f vz%+.2f dp%+.2f vxc%+.2f a2%+.2f q3%+.2f' % (h['t_lift'] - h['t_push'], h['l_vx'], h['l_vz'], h['l_dp'], h['l_vxc'], h['l_a2'], h['l_q3'])
        if 't_land' in h:
            s += ' | land +%d xc%+.3f xi%+.3f vxc%+.2f p%+.3f dp%+.2f a2%+.2f' % (h['t_land'] - h['t_lift'], h['d_xc'], h['d_xi'], h['d_vxc'], h['d_p'], h['d_dp'], h['d_a2'])
        if 'no_lift' in h:
            s += ' | NO LIFT'
        print(s)
    n = len(O)
    for t0 in range(max(0, n - trace_last), n, 2):
        o = O[t0]
        print('  %4d %-6s z %.3f p %+.3f dp %+.2f q %+.2f %+.2f %+.2f vx %+.2f vz %+.2f xc %+.3f xi %+.3f cop %+.3f fh %.3f a2 %+.2f u %s' % (
            t0, L['mode'][t0], o[0], o[1], o[7], o[2], o[3], o[4], o[5], o[6], L['xc'][t0], L['xi'][t0], L['cop'][t0], L['fh'][t0], L['a2'][t0], np.round(A[t0], 2)))
    return dict(ret=ret, x=x, steps=n, hops=hops, L=L)
