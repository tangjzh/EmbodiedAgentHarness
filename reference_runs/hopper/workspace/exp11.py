"""EXP11 (episode 11): real controller.policy (xi .20 -> .23 @330); from t600 push magnitude 0.5 (vs 0.6)."""
import numpy as np
import robot_api as robot
import controller
from runlib import parse, save_log

r = robot.reset()
o = robot.observe()
obs = np.asarray(o['observation'] if isinstance(o, dict) else o, float)
mem = {}
O = []; R = []; ret = 0.0; raw = {}
for t in range(1000):
    if t == 600 and 'P' in mem:
        mem['P']['push'] = 0.5
        print('t600: push -> 0.5')
    a = controller.policy(obs, t, mem)
    obs2, rew, term, trunc, raw = parse(robot.step(np.asarray(a, float).tolist()))
    O.append(obs); R.append(rew); ret += rew
    obs = obs2
    if term or trunc:
        print('TERMINAL at t', t, 'mode', mem.get('mode'))
        break
O = np.array(O)
save_log('exp11', {'obs': O, 'rew': np.array(R)})
print('steps', len(O), 'return %.2f' % ret, 'x_est %.3f' % (O[:, 5].sum() * 0.008), {k: v for k, v in raw.items() if k not in ('observation',)})
for h in mem.get('hops', []):
    print('  push t%4d xc%+.3f xi%+.3f vxc%+.2f a2%+.2f p%+.3f u%.2f' % (h['t'], h['xc'], h['xi'], h['vxc'], h['a2'], h['p'], h['push']))
for a0, b0 in [(100, 330), (330, 600), (600, 1000)]:
    b = min(b0, len(O))
    if b > a0:
        print('seg t[%d,%d) dx %.3f rate %.3f p[%+.3f,%+.3f] zmax %.3f' % (a0, b, O[a0:b, 5].sum() * 0.008, O[a0:b, 5].mean(), O[a0:b, 1].min(), O[a0:b, 1].max(), O[a0:b, 0].max()))
n = len(O)
for t0 in range(max(0, n - 16), n, 2):
    o = O[t0]
    print('  %4d z %.3f p %+.3f dp %+.2f q %+.2f %+.2f %+.2f vx %+.2f vz %+.2f' % (t0, o[0], o[1], o[7], o[2], o[3], o[4], o[5], o[6]))
