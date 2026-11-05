"""Summarise a development episode log: perception, phase transitions, landing, success onset, descent vs force."""
import json
import math
import sys

import numpy as np

tag = sys.argv[1]
L = [json.loads(l) for l in open(f'/workspace/data/{tag}/log.jsonl')]
P = [l for l in L if l['kind'] == 'perception'][0]
S = [l for l in L if l['kind'] == 'step']
E = [l for l in L if l['kind'] == 'end']
b, n = P['bolt'], P['nut']
print('episode', P.get('episode'), 'eef0', np.round(P['eef'], 4), 'psi0 %.1f deg' % math.degrees(P['psi']))
print('bolt xy', np.round(b['xy'], 6), 'tip %.5f' % b['ztip'], [(e['cam'], np.round(np.array(e['c']) * 1e3, 2).tolist(), round(e['ztip'], 5), e['n']) for e in b['ests']])
print('nut o_hand mm', np.round(np.array(n['o_hand']) * 1e3, 3), n['src'], [(e['cam'], np.round(np.array(e['o_hand']) * 1e3, 2).tolist(), np.round(np.array(e['widths']) * 1e3, 2).tolist(), e['good']) for e in n['ests']])
last = None
first_succ = None
for l in S:
    if l['phase'] != last:
        print('step %3d -> %-8s h %.2f mm exy %.2f mm nyaw %.1f' % (l['step'], l['phase'], l['h'] * 1e3, l['exy'] * 1e3, math.degrees(l['nyaw'])))
        last = l['phase']
    if l['succ'] and first_succ is None:
        first_succ = l
print('first success after step', first_succ['step'] if first_succ else None, 'h %.2f' % (first_succ['h'] * 1e3) if first_succ else '', 'nyaw %.1f' % math.degrees(first_succ['nyaw']) if first_succ else '')
print('end', E[0]['succ'] if E else None, 'final eef', E[0]['eef'] if E else None, 'final nyaw %.1f' % math.degrees(E[0]['nyaw']) if E else '')
# per 10 steps
st = np.array([l['step'] for l in S]); h = np.array([l['h'] for l in S]) * 1e3; ny = np.degrees([l['nyaw'] for l in S])
sz = np.array([l['s'][2] for l in S]); ex = np.array([l['exy'] for l in S]) * 1e3
for i in range(0, len(S), 10):
    j = slice(i, i + 10)
    print('%3d %-8s h %6.2f  nyaw %6.1f  s_z %5.2f exy %.2f' % (st[i], S[i]['phase'], h[j].mean(), ny[j].mean(), sz[j].mean(), ex[j].mean()))
