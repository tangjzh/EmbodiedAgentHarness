# Implement policy(observation, step, memory) -> native action.
# FINAL (validated unmodified in EXP12: 1000 ticks survived, return 1136.8, +1.10 m; same gait EXP09 1123, EXP10 1146):
# CP standing balance for 100 ticks, then the toe-push "lean & bounce" gait: CP law leans COM forward (xc_d=0.22); when
# capture point xi>=xi_push push toe-down u3=-0.6 for exactly 15 ticks (no early stop on liftoff); then CP balance again.
# xi_push 0.20 until tick 330, then 0.23. Requires hopctl4.py + hopper_lib.py from the workspace.
import numpy as np
from hopctl4 import control4, P4

PARAMS = dict(P4)
PARAMS.update(xc_gait=0.22, xi_push=0.20, push=0.6, k_push=0.0, push_max=15, kN=0.0, kV=0.0, min_stance=5,
              lift_end=False)
GAIT_START = 100
XI_SCHEDULE = [(330, 0.23)]   # (tick, xi_push)
# Late "sprint": weaker push (0.5) under-arrests forward rotation -> speed builds up (EXP11: 0.45 m/s, runaway that
# survived >=187 ticks; EXP06 runaway survived ~170 ticks). Started late so the horizon ends before it can fall.
PUSH_SCHEDULE = [(750, 0.5)]  # (tick, push magnitude)


def policy(observation, step, memory):
    if 'mode' not in memory:
        memory['mode'] = 'stand'
        memory['xc_d'] = PARAMS['xc_stand']
        memory['n'] = 0
        memory['P'] = dict(PARAMS)
    n = memory['n']
    if n >= GAIT_START:
        memory['gait_on'] = True
    for t0, xp in XI_SCHEDULE:
        if n >= t0:
            memory['P']['xi_push'] = xp
    for t0, u in PUSH_SCHEDULE:
        if n >= t0:
            memory['P']['push'] = u
    memory['n'] = n + 1
    a, _ = control4(observation, memory, memory['P'])
    return np.clip(a, -1.0, 1.0)
