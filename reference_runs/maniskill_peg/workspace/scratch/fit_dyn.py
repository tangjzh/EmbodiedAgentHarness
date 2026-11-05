import json, glob, numpy as np
from geom import rotvec, rotvec_to_R

def load_runs():
    out = []
    for fn in sorted(glob.glob('/workspace/data/runs/ep*_1*.json')):
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        out.append((fn.split('/')[-1], [e for e in d['log'] if 'phase' in e]))
    return out

rows = []
for name, L in load_runs():
    for k in range(2, len(L)):
        e0, e1, e2 = L[k-2], L[k-1], L[k]
        if e2['n'] != e1['n'] + 1 or e1['n'] != e0['n'] + 1:
            continue
        u = np.array(e2['a'][:3]) * 0.1
        up = np.array(e1['a'][:3]) * 0.1
        dprev = np.array(e1['p']) - np.array(e0['p'])
        dcur = np.array(e2['p']) - np.array(e1['p'])
        R0 = rotvec_to_R(np.array(e0['rv'])); R1 = rotvec_to_R(np.array(e1['rv'])); R2 = rotvec_to_R(np.array(e2['rv']))
        wprev = rotvec(R1 @ R0.T); wcur = rotvec(R2 @ R1.T)
        ur = -np.array(e2['a'][3:6]) * 0.1
        rows.append(dict(u=u, up=up, dp=dprev, dc=dcur, ur=ur, wp=wprev, wc=wcur, ph=e2['phase'], ep=name[:4]))
print(len(rows), 'transitions from', sorted(set(r['ep'] for r in rows)))
U = np.array([r['u'] for r in rows]); Dp = np.array([r['dp'] for r in rows]); Dc = np.array([r['dc'] for r in rows])
UP = np.array([r['up'] for r in rows])
for ax in range(3):
    A = np.c_[U[:, ax], Dp[:, ax]]
    coef, *_ = np.linalg.lstsq(A, Dc[:, ax], rcond=None)
    print('trans axis', ax, 'a %.3f b %.3f' % tuple(coef), 'rms %.4f' % np.sqrt(np.mean((A @ coef - Dc[:, ax])**2)))
A = np.c_[U.reshape(-1), Dp.reshape(-1)]
coef, *_ = np.linalg.lstsq(A, Dc.reshape(-1), rcond=None)
print('trans all a %.3f b %.3f rms %.4f' % (coef[0], coef[1], np.sqrt(np.mean((A @ coef - Dc.reshape(-1))**2))))
A = np.c_[U.reshape(-1), UP.reshape(-1), Dp.reshape(-1)]
coef3, *_ = np.linalg.lstsq(A, Dc.reshape(-1), rcond=None)
print('trans all a %.3f a1 %.3f b %.3f rms %.4f' % (coef3[0], coef3[1], coef3[2], np.sqrt(np.mean((A @ coef3 - Dc.reshape(-1))**2))))
UR = np.array([r['ur'] for r in rows]); WP = np.array([r['wp'] for r in rows]); WC = np.array([r['wc'] for r in rows])
A = np.c_[UR.reshape(-1), WP.reshape(-1)]
c, *_ = np.linalg.lstsq(A, WC.reshape(-1), rcond=None)
print('rot all a %.3f b %.3f rms %.4f' % (c[0], c[1], np.sqrt(np.mean((A @ c - WC.reshape(-1))**2))))
A = np.c_[UR[:, 2], WP[:, 2]]
c, *_ = np.linalg.lstsq(A, WC[:, 2], rcond=None)
print('rot z a %.3f b %.3f rms %.4f' % (c[0], c[1], np.sqrt(np.mean((A @ c - WC[:, 2])**2))))
# saturated rotation speeds
sat = np.linalg.norm(UR, axis=1) > 0.095
print('sat rot steps', sat.sum(), 'mean |w| %.4f max %.4f' % (np.linalg.norm(WC[sat], axis=1).mean(), np.linalg.norm(WC[sat], axis=1).max()))

print('---- free-motion subset (no ep5, phases approach/stage/look/look0/lift/descend)')
keep = [i for i, r in enumerate(rows) if r['ph'] in ('approach', 'stage', 'look', 'look0', 'lift', 'descend') and r['ep'] != 'ep5_']
U2, D2p, D2c, UP2 = U[keep], Dp[keep], Dc[keep], UP[keep]
for ax in range(3):
    A = np.c_[U2[:, ax], D2p[:, ax]]
    coef, *_ = np.linalg.lstsq(A, D2c[:, ax], rcond=None)
    print('trans axis', ax, 'a %.3f b %.3f' % tuple(coef), 'rms %.4f' % np.sqrt(np.mean((A @ coef - D2c[:, ax])**2)))
    A = np.c_[U2[:, ax], UP2[:, ax], D2p[:, ax]]
    coef, *_ = np.linalg.lstsq(A, D2c[:, ax], rcond=None)
    print('   2-lag a %.3f a1 %.3f b %.3f' % tuple(coef), 'rms %.4f' % np.sqrt(np.mean((A @ coef - D2c[:, ax])**2)))
A = np.c_[U2.reshape(-1), D2p.reshape(-1)]
coef, *_ = np.linalg.lstsq(A, D2c.reshape(-1), rcond=None)
print('all a %.3f b %.3f rms %.4f' % (coef[0], coef[1], np.sqrt(np.mean((A @ coef - D2c.reshape(-1))**2))))
