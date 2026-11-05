"""Development episode: spend one counted reset, then run the full frozen controller main() once.
Tag/cfg from scripts/CURRENT_TAG and scripts/cfg_<tag>.json; logs to /workspace/data/<tag>."""
import json
import os
import sys
import time

sys.path.insert(0, '/workspace')
import controller as C
import robot_api as robot

tag = open('/workspace/scripts/CURRENT_TAG').read().strip()
cfg_path = f'/workspace/scripts/cfg_{tag}.json'
cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
cfg.setdefault('log_dir', f'/workspace/data/{tag}')
cfg.setdefault('snap_every', 30)
cfg.setdefault('save_initial', True)
if os.path.exists(os.path.join(cfg['log_dir'], 'log.jsonl')):
    raise SystemExit(f'log for {tag} exists; refusing to overwrite')
o = robot.observe()
print('before: episode', o.get('episode'), 'active', o.get('active'), 'dev remaining', o.get('development_remaining'),
      'step', o.get('native_step'))
if o.get('active') and o.get('native_step') == 0 and not o.get('terminal'):
    print('episode already active at step 0 (counted by the operator restart): using it, no reset')
else:
    r = robot.reset()
    print('reset reply keys', list(r.keys()) if isinstance(r, dict) else type(r))
o = robot.observe()
print('after reset: episode', o.get('episode'), 'active', o.get('active'), 'dev remaining', o.get('development_remaining'), 'step', o.get('native_step'))
t0 = time.time()
out = C.main(cfg)
print('main returned after %.0fs' % (time.time() - t0))
print(json.dumps({k: out.get(k) for k in ('native_step', 'terminal', 'native_success', 'final_native_metric', 'episode')}))
