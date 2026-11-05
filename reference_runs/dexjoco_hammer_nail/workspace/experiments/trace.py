"""Summarize a run's step calls: rows, first/last targets, resulting states.

usage: trace.py <run_dir> [tool ...]
"""
import json
import sys

run = sys.argv[1]
tools = set(sys.argv[2:]) or {'dexjoco_step'}
req = None
for line in open(run + '/mcp.jsonl'):
    d = json.loads(line)
    if d.get('event') == 'mcp_request' and d.get('tool') in tools:
        a = (d.get('arguments') or {}).get('actions', [])
        req = (len(a), [round(v, 3) for v in a[0][:6]] if a else None,
               [round(v, 3) for v in a[-1][:6]] if a else None,
               [round(v, 2) for v in a[-1][6:]] if a else None)
    if d.get('event') == 'mcp_result' and d.get('tool') in tools and d.get('response_path'):
        r = json.load(open(d['response_path']))
        try:
            m = json.loads(r['content'][0]['text'])
        except (KeyError, ValueError, IndexError):
            print(d['response_path'], 'unparsable')
            continue
        st = m.get('state') or m.get('end_state')
        print(d['response_path'].split('/')[-2], d.get('tool'), req[0] if req else None, 'first', req[1] if req else None,
              'last', req[2] if req else None, 'hand', req[3] if req else None)
        if st:
            print('     -> step', m.get('step_id'), 'state', [round(v, 3) for v in st[:7]], 'hand',
                  [round(v, 2) for v in st[7:]], 'tc', (m.get('transition') or {}).get('task_complete'))
