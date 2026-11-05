"""Pump-only sweep on my model (no dev episodes): which pump gains get the ball above the rim fastest from low starts?"""
import itertools
import controller as C


def pump_metric(par, start, T=350):
    s = start
    best, first = -9.0, None
    for k in range(T):
        ux, uz = C._pump(s, par)
        s = C._tick(s, ux, uz)
        lx, lz = C._local(s)
        best = max(best, lz)
        if first is None and lz > 0.06:
            first = k
    return best, first


starts = [(0.0, 0.0, bx, bz, 0.0, 0.0, 0.0, 0.0) for bx, bz in
          [(0.0, -0.008), (0.1, 0.0), (-0.1, 0.0), (0.105, 0.2), (-0.084, 0.206), (0.057, 0.256), (-0.004, 0.209)]]
rows = []
for lam, kz, T0 in itertools.product([-8.0, -4.0, 0.0, 3.0, 5.5, 8.0, 12.0], [0.0, 0.1, 0.25, 0.5], [10.0, 14.0, 20.0, 30.0]):
    if kz == 0.0 and T0 != 14.0:
        continue
    par = dict(C.PAR, lam=lam, kz=kz, T0=T0)
    res = [pump_metric(par, s) for s in starts]
    n_ok = sum(1 for b, f in res if f is not None)
    t_max = max((f if f is not None else 999) for b, f in res)
    worst = min(b for b, f in res)
    rows.append((n_ok, -t_max, round(worst, 3), lam, kz, T0))
rows.sort(reverse=True)
print("n_ok/7  -t_max  worst_max_lz  lam kz T0")
for r in rows[:14]:
    print(r)
print("...worst configs")
for r in rows[-4:]:
    print(r)
print("current PAR result:", [pump_metric(C.PAR, s) for s in starts])
