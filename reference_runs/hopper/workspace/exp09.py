"""EXP09 (episode 9): push always lasts full N=15 ticks (lift_end=False); xi_push 0.20 -> 0.23 at t=330; u=0.6."""
import numpy as np
import robot_api as robot
from hopctl4 import control4, P4
from runctl import run

r = robot.reset()
P = dict(P4)
P.update(xc_gait=0.22, xi_push=0.20, push=0.6, k_push=0.0, push_max=15, kN=0.0, kV=0.0, min_stance=5, lift_end=False)


def schedule(t, mem, P):
    if t == 330:
        P['xi_push'] = 0.23


res = run(control4, P, 'exp09', gait_start=100, trace_last=24, schedule=schedule, verbose_hops=30)
O = np.array(res['L']['obs'])
for a in range(100, len(O), 150):
    b = min(len(O), a + 150)
    print('win t[%d,%d) dx %.3f rate %.3f p[%+.3f,%+.3f] zmax %.3f' % (a, b, O[a:b, 5].sum() * 0.008, O[a:b, 5].mean(), O[a:b, 1].min(), O[a:b, 1].max(), O[a:b, 0].max()))
