"""EXP05 (episode 5): stand 100 ticks, then continuous toe-lean bunny-hop gait (xc_d=0.22, push at xi>=0.20, u3=-0.6)."""
import robot_api as robot
from hopctl4 import control4, P4
from runctl import run

r = robot.reset()
P = dict(P4)
P.update(xc_gait=0.22, xi_push=0.20, push=0.6, k_push=0.0, push_max=15, min_stance=5)
res = run(control4, P, 'exp05', gait_start=100, trace_last=80)
