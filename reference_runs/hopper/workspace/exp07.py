"""EXP07 (episode 7): adaptive push: N = clip(12 + (xc-0.12)/0.01, 8, 20) ticks, u = 0.6 + 4*max(0, xc-0.16)."""
import numpy as np
import robot_api as robot
from hopctl4 import control4, P4
from runctl import run

r = robot.reset()
P = dict(P4)
P.update(xc_gait=0.22, xi_push=0.23, push=0.6, k_push=4.0, xc_hard=0.16, push_max=12, kN=1.0, xc_ref=0.12,
         N_lo=8, N_hi=20, min_stance=5)
res = run(control4, P, 'exp07', gait_start=100, trace_last=30)
O = np.array(res['L']['obs'])
for a in range(100, len(O), 150):
    b = min(len(O), a + 150)
    print('win t[%d,%d) dx %.3f rate %.3f p[%+.3f,%+.3f] zmax %.3f' % (a, b, O[a:b, 5].sum() * 0.008, O[a:b, 5].mean(), O[a:b, 1].min(), O[a:b, 1].max(), O[a:b, 0].max()))
H = res['hops']
print('push N/u stats:', [(h['t_push'], round(h['xc'], 3), h.get('push')) for h in H[-12:]])
