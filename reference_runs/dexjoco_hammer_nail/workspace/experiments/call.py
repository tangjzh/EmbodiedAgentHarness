"""Call an MCP tool from Python with rows from a JSON file; save metadata and images.

usage: python3 experiments/call.py <tool> [rows.json] [tag]
"""
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '.')
from api.runtime import Context

tool = sys.argv[1]
rows_file = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '-' else None
tag = sys.argv[3] if len(sys.argv) > 3 else tool
out = Path('output/calls')
out.mkdir(parents=True, exist_ok=True)
args = {}
if rows_file:
    args['actions'] = json.load(open(rows_file))
t0 = time.time()
with Context() as ctx:
    reply = ctx.call(tool, **args)
dt = time.time() - t0
meta = None
k = 0
for b in reply.get('content', []):
    if b.get('type') == 'text' and meta is None:
        try:
            meta = json.loads(b['text'])
        except ValueError:
            meta = {'text': b['text']}
    elif b.get('type') == 'image':
        k += 1
        (out / f'{tag}_img{k}.png').write_bytes(base64.b64decode(b['data']))
json.dump(meta, open(out / f'{tag}.json', 'w'), indent=1)
drop = ('camera_intrinsics', 'camera_poses', 'depth', 'attachments', 'transition')
short = {kk: v for kk, v in (meta or {}).items() if kk not in drop}
if isinstance(meta, dict) and 'transition' in meta:
    tr = meta['transition']
    short['transition'] = {kk: v for kk, v in tr.items() if kk != 'steps'}
    st = tr.get('steps', [])
    short['n_steps'] = len(st)
    if st:
        short['last_step'] = {kk: v for kk, v in st[-1].items() if kk != 'executed_action'}
if isinstance(meta, dict) and 'state' in meta:
    json.dump(meta['state'], open('output/state.json', 'w'))
s = json.dumps(short)
print(f'{dt:.1f}s', s[:2500])
print('images saved:', k)
