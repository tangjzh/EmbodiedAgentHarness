"""EXP06 (episode 6): in-episode parameter sweep of the EXP05 toe-push gait (segments ordered by risk)."""
import numpy as np
import robot_api as robot
from hopctl4 import control4, P4
from runctl import run

r = robot.reset()
P = dict(P4)
P.update(xc_gait=0.22, xi_push=0.20, push=0.6, k_push=0.0, push_max=15, min_stance=5)
SEG = [(100, dict(xi_push=0.20, push=0.6, push_max=15)),
       (330, dict(xi_push=0.23, push=0.6, push_max=15)),
       (560, dict(xi_push=0.23, push=0.6, push_max=8)),
       (780, dict(xi_push=0.26, push=0.7, push_max=10))]


def schedule(t, mem, P):
    for t0, d in SEG:
        if t == t0:
            P.update(d)
            print('t', t, 'set', d)


res = run(control4, P, 'exp06', gait_start=100, trace_last=40, schedule=schedule)
O = np.array(res['L']['obs'])
bounds = [s[0] for s in SEG] + [len(O)]
for i in range(len(SEG)):
    a, b = bounds[i], min(bounds[i + 1], len(O))
    if b > a:
        print('SEG %d t[%d,%d) dx %.3f rate %.3f m/s  p[%+.3f,%+.3f] zmax %.3f' % (
            i, a, b, O[a:b, 5].sum() * 0.008, O[a:b, 5].mean(), O[a:b, 1].min(), O[a:b, 1].max(), O[a:b, 0].max()))
