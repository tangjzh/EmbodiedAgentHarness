"""Development runner: call controller.main with a cfg from scripts/cfg_<tag>.json, logging to data/<tag>."""
import json
import os
import sys

sys.path.insert(0, '/workspace')
import controller as C

tag = open('/workspace/scripts/CURRENT_TAG').read().strip()
cfg = json.load(open(f'/workspace/scripts/cfg_{tag}.json')) if os.path.exists(f'/workspace/scripts/cfg_{tag}.json') else {}
cfg.setdefault('log_dir', f'/workspace/data/{tag}')
cfg.setdefault('snap_every', 30)
out = C.main(cfg)
print(json.dumps({k: out.get(k) for k in ('native_step', 'terminal', 'native_success', 'final_native_metric', 'episode')}))
