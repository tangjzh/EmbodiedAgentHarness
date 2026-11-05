# Dev harness: spend one development reset, save initial frames, run the frozen controller main().
import sys, json, time
import numpy as np
sys.path.insert(0, '/workspace')
import robot_api as robot
res = robot.reset()
obs = robot.observe()
ep = obs.get('episode')
cams = robot.cameras()
np.savez_compressed(f'/workspace/data/ep{ep}_init.npz', **cams)
json.dump(obs, open(f'/workspace/data/ep{ep}_init_obs.json', 'w'))
print('episode', ep, 'remaining', obs.get('development_remaining'), 'step', obs.get('native_step'))
import controller
controller.main()
obs = robot.observe()
try:
    cams = robot.cameras()
    np.savez_compressed(f'/workspace/data/ep{ep}_final.npz', **cams)
except Exception as e:
    print('final cams failed', e)
json.dump(obs, open(f'/workspace/data/ep{ep}_final_obs.json', 'w'))
print('final step', obs.get('native_step'), 'success', obs.get('native_success'), 'return', obs.get('native_return'))
