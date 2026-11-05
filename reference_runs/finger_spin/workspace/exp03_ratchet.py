import numpy as np, robot_api as robot, json, sys, os
import ctrl
np.set_printoptions(precision=3, suppress=True, linewidth=180)

N = int(os.environ.get('N', '250'))
P = dict(ctrl.DEFAULT)
mem = {}
o = robot.observe()
obs = o['observation'] if 'observation' in o else o
rows = []
ret0 = o.get('native_return', 0.0)
for i in range(N):
    a, info = ctrl.act(obs, mem, P)
    r = robot.step([float(a[0]), float(a[1])], 1)
    obs = r['observation']
    q, dq, th, w = ctrl.decode(obs)
    rew = r['rewards'][-1] if r.get('rewards') else np.nan
    rows.append([r['step'], q[0], q[1], dq[0], dq[1], th, w, obs['touch'][0], obs['touch'][1], a[0], a[1],
                 rew, np.rad2deg(info['psi']), info['p'], {'wait': 0, 'push': 1, 'nudge': 2, 'strike': 3}[info['mode']]])
    if r.get('terminal'):
        break
arr = np.array(rows)
np.save('logs/exp03.npy', arr)
print('cols: step q1 q2 dq1 dq2 th w t0 t1 a0 a1 rew psi p(at decision) mode')
for row in arr[::1][:160]:
    print(row)
print('return now', r['native_return'], 'gained', r['native_return'] - ret0, 'mean w', arr[:, 6].mean(),
      'frac w<=-15', (arr[:, 6] <= -15).mean())
