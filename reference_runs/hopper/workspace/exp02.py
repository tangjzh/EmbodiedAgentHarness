"""EXP02 (episode 2): posture-consistent stance. stand -> squat -> single hops -> continuous in-place hopping."""
import numpy as np
import robot_api as robot
from hopctl import control, DEFAULT_P
from runlib import parse, save_log, new_log, append, print_segments

# NOTE (continuation): simulator already at episode 2 tick 0 -> no reset here.
P = dict(DEFAULT_P)
mem = {'mode': 'stand', 'q2_ref': 0.0, 'p_d': 0.0}
obs = np.asarray(robot.observe(), float)
log = new_log()
hops = [dict(u2_thrust=0.5, thrust_ankle='posture'), dict(u2_thrust=0.8, thrust_ankle='posture'),
        dict(u2_thrust=0.6, thrust_ankle='hold')]
hop_i = 0
t_stand = 0
ret = 0.0
for t in range(1000):
    if 60 <= t < 160:
        mem['q2_ref'] = -0.25 * (1 - np.cos(2 * np.pi * (t - 60) / 100.0))
    if mem['mode'] == 'stand':
        t_stand += 1
        if t >= 170:
            mem['q2_ref'] = min(0.0, mem.get('q2_ref', 0.0) + 0.005)
            if t_stand > 70:
                if hop_i < len(hops):
                    mem.update(hops[hop_i]); print(f't={t} hop {hop_i}: {hops[hop_i]}')
                    hop_i += 1
                elif not mem.get('continuous', False):
                    mem.update(dict(u2_thrust=0.7, thrust_ankle='posture', continuous=True))
                    print(f't={t} start continuous hopping')
                mem['q2_start'] = mem['q2_ref']; mem['mode'] = 'crouch'; mem['t_mode'] = 0
                t_stand = 0
    else:
        t_stand = 0
    a, st = control(obs, mem, P)
    obs2, rew, term, trunc, raw = parse(robot.step(a.tolist()))
    ret += rew
    append(log, obs, a, rew, mem['mode'], st)
    obs = obs2
    if term or trunc:
        print('TERMINAL at t', t)
        break
save_log('exp02', log)
O = np.array(log['obs'])
print('steps', len(O), 'return', round(ret, 2), 'sum vx*dt', round(O[:, 5].sum() * 0.008, 3))
print_segments(log)
for t0 in [0, 59, 110, 159]:
    s = O[t0]
    print(t0, 'z %.3f p %.3f q %.3f %.3f %.3f a1 %.3f a2 %.3f xc %.3f act %s' % (
        s[0], s[1], s[2], s[3], s[4], log['a1'][t0], log['a2'][t0], log['xc'][t0], np.round(log['act'][t0], 3)))
print('final obs', np.round(O[-1], 3), 'raw', {k: v for k, v in raw.items() if k != 'observation'})
# per-flight analytics
M = log['mode']
for t in range(1, len(M)):
    if M[t] == 'flight' and M[t - 1] != 'flight':
        t1 = t
        while t1 < len(M) and M[t1] == 'flight':
            t1 += 1
        o0 = O[t]; o1 = O[min(t1, len(O) - 1)]
        print(f'FLIGHT t{t}-{t1} n{t1 - t}: takeoff vx {o0[5]:+.2f} vz {o0[6]:+.2f} p {o0[1]:+.3f} dp {o0[7]:+.2f} q {np.round(o0[2:5], 2)}'
              f' | land p {o1[1]:+.3f} dp {o1[7]:+.2f} vx {o1[5]:+.2f} vz {o1[6]:+.2f} q {np.round(o1[2:5], 2)} a2 {log["a2"][min(t1, len(O) - 1)]:+.2f}'
              f' maxfoot {max(log["foot_h"][t:t1]):.3f}')
for t0 in range(160, len(O), 10):
    s = O[t0]
    print(t0, M[t0][:6], 'z %.3f p %+.3f q %+.2f %+.2f %+.2f vx %+.2f vz %+.2f dp %+.2f a2 %+.2f xc %+.3f fh %.3f act %s' % (
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], log['a2'][t0], log['xc'][t0], log['foot_h'][t0], np.round(log['act'][t0], 2)))
