import numpy as np, robot_api as robot
from lib import *

np.set_printoptions(precision=3, suppress=True, linewidth=160)
rec = Rec(robot, 'exp02_strike')


def st():
    return unpack(rec.obs())


def goto(qd, tag, tol=0.04, maxn=40, K=(2.5, 5.0)):
    qd = np.asarray(qd, float)
    for i in range(maxn):
        q, dq, th, w, t = st()
        if np.all(np.abs(qd - q) < tol) and np.all(np.abs(dq) < 1.0):
            return i
        e = qd - q
        rec.step([K[0] * e[0], K[1] * e[1]], tag)
    return maxn


def strike(r_arc, tag):
    # ready pose 30 deg below centre line on arc radius r_arc about B
    al = np.deg2rad(-30)
    p_ready = B + r_arc * np.array([np.cos(al), np.sin(al)])
    qr = ik(p_ready, +1)
    q, dq, th, w, t = st()
    goto([q[0], 1.6], tag)            # fold
    goto([qr[0], 1.6], tag)           # swing down folded (safe radius)
    goto(qr, tag)                     # unfold to ready
    q, dq, th, w, t = st()
    print(f'ready r_arc={r_arc} q={q} th={th:.3f} w={w:.2f} gap={gap(q, th):.3f}')
    n0 = len(rec.rows)
    for i in range(10):               # full-speed upward swing holding q2
        q, dq, th, w, t = st()
        if q[0] > np.arctan2(0.16, r_arc * 0.87) + 0.25:
            break
        rec.step([1.0, 5.0 * (qr[1] - q[1])], tag + 1)
    goto([st()[0][0], 1.6], tag + 2)  # fold away
    seg = np.array(rec.rows[n0 - 1:])
    print('strike segment: q1 q2 dq1 dq2 th w t0 t1')
    for r in seg[:14]:
        print(r[:8], f'gap={gap(r[:2], r[4]):.3f}')
    # coast: hold folded far pose
    for i in range(55):
        q, dq, th, w, t = st()
        e = np.array([0.3, 1.6]) - q
        rec.step([2.5 * e[0], 5 * e[1]], tag + 3)
    arr = np.array(rec.rows)
    w_hist = arr[n0:, 5]
    print('w history (every 2):', w_hist[::2])


strike(0.32, 10)
strike(0.30, 20)
strike(0.31, 30)
arr = rec.save()
print('total steps so far', len(arr) + 68, 'return', rec.last.get('native_return'))
