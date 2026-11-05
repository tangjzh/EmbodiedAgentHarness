"""Offline geometry: free-space map of fingertip (on arc radius rB about B, angle psi) vs spinner angle th.
No robot steps. '#' = overlap with bar, '.' = free."""
import numpy as np
from lib import *

def q_of(rB, psi):
    return ik(B + rB * np.array([np.cos(psi), np.sin(psi)]), +1)

ths = np.deg2rad(np.arange(0, 180, 7.5))
psis = np.deg2rad(np.arange(-36, 37, 3))
for rB in (0.27, 0.29, 0.30, 0.31):
    q2 = q_of(rB, 0)[1]
    print(f'rB={rB} q2={q2:.3f}  rows: th(deg, tip-end polar = th+90); cols psi from -36 to 36 step 3')
    for th in ths:
        row = ''.join('#' if gap(q_of(rB, p), th) < 0 else '.' for p in psis)
        print(f'{np.rad2deg(th):6.1f} {row}')
try:
    import mujoco  # only checking availability; not used
    print('mujoco importable (NOT used)')
except Exception as e:
    print('no mujoco:', type(e).__name__)
