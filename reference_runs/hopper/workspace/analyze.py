"""Offline analysis of saved logs (no physics advance)."""
import os, sys, pathlib
import numpy as np
print('workspace:', sorted(os.listdir('.')))
if os.path.isdir('logs'):
    for f in sorted(os.listdir('logs')):
        p = os.path.join('logs', f)
        print('  log', f, os.path.getsize(p), 'bytes')
for n in range(3, 21):
    p = pathlib.Path('exp%02d.py' % n)
    if p.exists():
        txt = p.read_text()
        print('exp%02d.py' % n, len(txt), 'chars', '(stub)' if txt.strip() == '# stub' else '')
for n in ['hopctl4.py', 'hopctl5.py', 'runctl.py']:
    print(n, len(pathlib.Path(n).read_text()))
if os.path.exists('logs/exp03.npz'):
    L = np.load('logs/exp03.npz', allow_pickle=True)
    O = L['obs']; A = L['act']; M = L['mode']; PH = L['phase']
    print('exp03 steps', len(O))
    last = None
    for t in range(len(O)):
        key = (str(M[t]), str(PH[t]))
        if key != last:
            o = O[t]
            print('  t%4d mode %-6s prog %-6s z %.3f p %+.3f dp %+.2f q %+.2f %+.2f %+.2f vx %+.2f vz %+.2f xc %+.3f fh %.3f u %s' % (
                t, key[0], key[1], o[0], o[1], o[7], o[2], o[3], o[4], o[5], o[6], L['xc'][t], L['fh'][t], np.round(A[t], 2)))
            last = key
    for t in range(max(0, len(O) - 40), len(O), 2):
        o = O[t]
        print('  END t%4d %-6s z %.3f p %+.3f dp %+.2f q %+.2f %+.2f %+.2f vx %+.2f vz %+.2f xc %+.3f fh %.3f u %s' % (
            t, M[t], o[0], o[1], o[7], o[2], o[3], o[4], o[5], o[6], L['xc'][t], L['fh'][t], np.round(A[t], 2)))
    print('x_est total', O[:, 5].sum() * 0.008)
