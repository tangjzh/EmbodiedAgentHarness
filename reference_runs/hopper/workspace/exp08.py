"""EXP08 (episode 8): candidate: xi_push 0.20 -> 0.23 at t=330; u=0.6; N = clip(15 + 0.6*(xc-.12)/.01 + 1.0*(vxc-.45)/.1, 10, 20)."""
import numpy as np
import robot_api as robot
from hopctl4 import control4, P4
from runctl import run

r = robot.reset()
P = dict(P4)
P.update(xc_gait=0.22, xi_push=0.20, push=0.6, k_push=0.0, push_max=15, kN=0.6, xc_ref=0.12, kV=1.0, vxc_ref=0.45,
         N_lo=10, N_hi=20, min_stance=5)


def schedule(t, mem, P):
    if t == 330:
        P['xi_push'] = 0.23


res = run(control4, P, 'exp08', gait_start=100, trace_last=24, schedule=schedule, verbose_hops=30)
O = np.array(res['L']['obs'])
for a in range(100, len(O), 150):
    b = min(len(O), a + 150)
    print('win t[%d,%d) dx %.3f rate %.3f p[%+.3f,%+.3f] zmax %.3f' % (a, b, O[a:b, 5].sum() * 0.008, O[a:b, 5].mean(), O[a:b, 1].min(), O[a:b, 1].max(), O[a:b, 0].max()))
