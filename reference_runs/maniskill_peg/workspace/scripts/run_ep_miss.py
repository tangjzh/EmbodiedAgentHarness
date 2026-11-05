# Dev harness (fault injection): spend one development reset and run the unmodified controller main(), but make the
# FIRST grasp miss on purpose (grasp point moved 4.5 cm beyond the peg end farther from the box) so the real
# regrasp -> transport -> insert fallback path is exercised.  Only this script patches Runner; controller.py is untouched.
import sys, json
import numpy as np
sys.path.insert(0, '/workspace')
import robot_api as robot
from motion import reach_excess

res = robot.reset()
obs = robot.observe()
ep = obs.get('episode')
cams = robot.cameras()
np.savez_compressed(f'/workspace/data/ep{ep}_init.npz', **cams)
json.dump(obs, open(f'/workspace/data/ep{ep}_init_obs.json', 'w'))
print('episode', ep, 'remaining', obs.get('development_remaining'), 'step', obs.get('native_step'))
import controller

state = dict(miss=True, q=None)
orig_target = controller.Runner.grasp_target
orig_regrasp = controller.Runner.regrasp


def miss_target(self):
    gxy, Rg = orig_target(self)
    if not state['miss']:
        return gxy, Rg
    ax = np.array([np.cos(self.peg['yaw']), np.sin(self.peg['yaw'])])
    box = None if self.hole is None else np.asarray(self.hole)[:2]
    best = None
    for sgn in (-1.0, 1.0):
        q = self.peg['c'] + sgn * (self.L + 0.045) * ax
        if reach_excess(np.r_[q, 0.02], 0.03) > 0:
            continue
        d = 1.0 if box is None else float(np.linalg.norm(q - box))
        if best is None or d > best[0]:
            best = (d, q)
    if best is None:
        return gxy, Rg
    if state['q'] is None:
        print('[inject] forced miss: grasp point', np.round(best[1], 4), 'instead of', np.round(gxy, 4))
    state['q'] = best[1]
    return best[1], Rg


def regrasp(self):
    state['miss'] = False
    self.info['forced_miss_at'] = None if state['q'] is None else state['q']
    return orig_regrasp(self)


controller.Runner.grasp_target = miss_target
controller.Runner.regrasp = regrasp
controller.main()
obs = robot.observe()
try:
    cams = robot.cameras()
    np.savez_compressed(f'/workspace/data/ep{ep}_final.npz', **cams)
except Exception as e:
    print('final cams failed', e)
json.dump(obs, open(f'/workspace/data/ep{ep}_final_obs.json', 'w'))
print('final step', obs.get('native_step'), 'success', obs.get('native_success'), 'return', obs.get('native_return'))
