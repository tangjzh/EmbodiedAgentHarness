"""Development: spend a reset, then run the frozen controller once and keep its log."""
import shutil, sys, os, json
sys.path.insert(0, '/workspace')
import robot_api as robot
obs = robot.observe()
if not obs.get('active') or obs.get('terminal') or obs.get('native_step', 0) > 0:
    robot.reset()
import controller
controller.main()
n = robot.observe()['episode']
shutil.copy(controller.LOG, f'/workspace/tools/ep{n}_log.jsonl')
for suf in ('init.npz', 'init_obs.json'):
    if os.path.exists(f'/tmp/gearmesh_{suf}'):
        shutil.copy(f'/tmp/gearmesh_{suf}', f'/workspace/tools/ep{n}_{suf}')
L = [json.loads(l) for l in open(controller.LOG)]
first = next((r['t'] for r in L if r.get('succ')), None)
print('episode', n, 'first success step', first)
