import numpy as np
def sim(a, b, kr, kd, w0, T=1.0, n=50, frac=0.0, umax=0.1):
    th, w, up = 0.0, 0.0, 0.0
    tr = []
    for k in range(n):
        e = T - th
        u = kr * e - kd * min(1.0, abs(w) / w0) * w
        u = float(np.clip(u, -umax, umax))
        ue = (1 - frac) * u + frac * up
        w = a * ue + b * w
        th += w; up = u
        tr.append(th)
    tr = np.array(tr); err = np.abs(tr - T)
    conv = next((k + 1 for k in range(n) if np.all(err[k:] < 0.008)), None)
    return conv, tr.max() - T
for (a, b) in [(0.243, 0.336), (0.187, 0.482), (0.2, 0.6)]:
    for frac in (0.0, 0.3):
        out = []
        for (kr, kd, w0) in [(2, 0, 1), (2, 1, 0.02), (2, 1.5, 0.02), (2, 1, 0.03), (2.5, 1.5, 0.02)]:
            for T in (1.0, 0.1):
                c, ov = sim(a, b, kr, kd, w0, T=T, frac=frac)
                out.append('kr%.1f kd%.1f w0 %.2f T%.1f c=%s ov=%.4f' % (kr, kd, w0, T, c, ov))
        print((a, b), 'frac', frac)
        for o in out: print('   ', o)
