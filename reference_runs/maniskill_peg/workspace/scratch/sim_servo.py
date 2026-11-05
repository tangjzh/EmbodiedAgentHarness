import numpy as np
# per-axis models fitted on free-motion logs: v_k = a u_k + a1 u_{k-1} + b v_{k-1}
MODELS = {'1lag': (0.244, 0.0, 0.406), 'y2lag': (0.173, 0.124, 0.253), 'z2lag': (0.208, 0.142, 0.092),
          'x1lag': (0.313, 0.0, 0.252), 'ylag1': (0.219, 0.0, 0.474)}
def sim(model, ctrl, T=0.3, n=25):
    a, a1, b = MODELS[model]
    p, v, up = 0.0, 0.0, 0.0
    traj = []
    for k in range(n):
        u = ctrl(T - p, v, up)
        u = float(np.clip(u, -0.1, 0.1))
        v = a * u + a1 * up + b * v
        p += v; up = u
        traj.append(p)
    traj = np.array(traj)
    err = np.abs(traj - T)
    conv = next((k + 1 for k in range(n) if np.all(err[k:] < 0.002)), None)
    return conv, traj.max() - T, traj
def pctrl(kp, kd=0.0):
    return lambda e, v, up: kp * e - kd * v
for m in MODELS:
    print(m)
    for kp, kd in [(2, 0), (1.5, 0), (2, 1), (2, 2), (2.5, 2), (3, 3), (2, 3)]:
        c, ov, tr = sim(m, pctrl(kp, kd))
        print('  kp %.1f kd %.1f conv %s overshoot %.4f' % (kp, kd, c, ov))

print('--- robustness: extra delay / gain variations, kp3 kd1.5 vs kp2 kd1')
def sim2(a, a1, b, ctrl, delay=0, T=0.3, n=30, T2=None):
    p, v = 0.0, 0.0
    us = [0.0] * (delay + 2)
    traj = []
    for k in range(n):
        u = float(np.clip(ctrl(T - p, v, 0), -0.1, 0.1))
        us.append(u)
        ue, ue1 = us[-1 - delay], us[-2 - delay]
        v = a * ue + a1 * ue1 + b * v
        p += v
        traj.append(p)
    traj = np.array(traj); err = np.abs(traj - T)
    conv = next((k + 1 for k in range(n) if np.all(err[k:] < 0.002)), None)
    tail = np.ptp(traj[-8:])
    return conv, traj.max() - T, tail
for (a, a1, b) in [(0.244, 0, 0.406), (0.173, 0.124, 0.253), (0.3, 0, 0.55), (0.35, 0.1, 0.4)]:
    for delay in (0, 1):
        out = []
        for kp, kd in [(3, 1.5), (2, 0), (2, 1), (2, 1.5), (1.5, 0.75)]:
            c, ov, tail = sim2(a, a1, b, pctrl(kp, kd), delay)
            out.append('kp%.1f/kd%.2f c=%s ov=%.3f tail=%.4f' % (kp, kd, c, ov, tail))
        print((a, a1, b), 'delay', delay, ' | '.join(out))

print('--- velocity-scheduled damping kd_eff = kd*min(1,|v|/v0)')
def sched(kp, kd, v0):
    return lambda e, v, up: kp * e - kd * min(1.0, abs(v) / v0) * v
def sim3(a, a1, b, ctrl, frac=0.0, T=0.3, n=30):
    # frac: fraction of the action that acts one step late (fractional delay)
    p, v, u_prev = 0.0, 0.0, 0.0
    traj = []
    for k in range(n):
        u = float(np.clip(ctrl(T - p, v, 0), -0.1, 0.1))
        ue = (1 - frac) * u + frac * u_prev
        ue1 = (1 - frac) * u_prev
        v = a * ue + a1 * ue1 + b * v
        p += v; u_prev = u
        traj.append(p)
    traj = np.array(traj); err = np.abs(traj - T)
    conv = next((k + 1 for k in range(n) if np.all(err[k:] < 0.002)), None)
    return conv, traj.max() - T, np.ptp(traj[-8:])
for (a, a1, b) in [(0.244, 0, 0.406), (0.173, 0.124, 0.253), (0.3, 0, 0.55), (0.35, 0.1, 0.4)]:
    for frac in (0.0, 0.3, 0.5):
        out = []
        for name, c in [('p2', pctrl(2, 0)), ('p2d1', pctrl(2, 1)), ('s1/.02', sched(2, 1, 0.02)), ('s1.5/.02', sched(2, 1.5, 0.02)), ('s1/.03', sched(2, 1, 0.03))]:
            for T in (0.3, 0.08):
                cv, ov, tail = sim3(a, a1, b, c, frac, T=T)
                out.append('%s T%.2f c=%s ov=%.3f' % (name, T, cv, ov))
        print((a, a1, b), 'frac', frac); print('   ', ' | '.join(out[:6])); print('   ', ' | '.join(out[6:]))

print('--- z2lag model with scheduled damping')
for frac in (0.0, 0.3):
    for T in (0.06, 0.3):
        out = []
        for name, c in [('p2', pctrl(2, 0)), ('s1.5/.02', sched(2, 1.5, 0.02))]:
            cv, ov, tail = sim3(0.208, 0.142, 0.092, c, frac, T=T)
            out.append('%s c=%s ov=%.4f tail=%.4f' % (name, cv, ov, tail))
        print('frac', frac, 'T', T, ' | '.join(out))
