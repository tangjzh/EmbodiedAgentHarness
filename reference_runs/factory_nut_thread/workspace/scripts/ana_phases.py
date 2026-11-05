"""Phase transitions, re-fits, errors and success of one development episode. usage: ana_phases.py <tag>"""
import json
import sys

import numpy as np

tag = sys.argv[1]
L = [json.loads(l) for l in open(f'/workspace/data/{tag}/log.jsonl')]
P = [l for l in L if l['kind'] == 'perception'][0]
print(tag, 'psi0 %.1f' % np.degrees(P['psi']), 'bolt', np.round(np.array(P['bolt']['xy']) * 1e3, 3), 'tip %.5f' % P['bolt']['ztip'],
      'zb %.2f' % (P['zb'] * 1e3), 'o_hand', np.round(np.array(P['nut']['o_hand']) * 1e3, 2), 'bias', P['cfg'].get('axis_bias'))
for l in L:
    if l['kind'] == 'reperceive':
        b = l['bolt'] or {}
        print('  reperceive out', {k: (np.round(np.array(v) * 1e3, 3).tolist() if v is not None else None) for k, v in l['out'].items()},
              [(e['cam'], np.round(np.array(e['c']) * 1e3, 3).tolist(), round(e['r'] * 1e3, 2)) for e in b.get('ests', [])])
    elif l['kind'] == 'error':
        print('  ERROR', l)
S = [l for l in L if l['kind'] == 'step']
ph = None
for l in S:
    if l['phase'] != ph:
        print('  %3d %-8s h %+.2f exy %.2f nyaw %6.1f so %s ax %s' % (l['step'], l['phase'], l['h'] * 1e3, l['exy'] * 1e3, np.degrees(l['nyaw']),
              np.round(np.array(l['so']) * 1e3, 2), np.round(np.array(l['ax']) * 1e3, 3)))
        ph = l['phase']
fs = [l for l in S if l['succ']]
print('  first succ', (fs[0]['step'], round(fs[0]['h'] * 1e3, 2), round(np.degrees(fs[0]['nyaw']), 1)) if fs else None)
for k in (100, 200, 300, 400, 440):
    c = [l for l in S if l['step'] == k]
    if c:
        print('  step %d %s h %+.2f nyaw %.1f exy %.2f' % (k, c[0]['phase'], c[0]['h'] * 1e3, np.degrees(c[0]['nyaw']), c[0]['exy'] * 1e3))
E = [l for l in L if l['kind'] == 'end']
if E:
    print('  end', E[0]['step'], 'succ', E[0]['succ'], 'nyaw %.1f' % np.degrees(E[0]['nyaw']))
