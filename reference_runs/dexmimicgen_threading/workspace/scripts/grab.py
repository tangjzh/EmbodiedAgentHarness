import sys, json, time
import numpy as np
import robot_api as robot

tag = sys.argv[1] if len(sys.argv) > 1 else 'grab'
obs = robot.observe()
cams = robot.cameras()
for k, v in cams.items():
    print(k, v.shape, v.dtype, float(np.nanmin(v)), float(np.nanmax(v)))
np.savez_compressed(f'/workspace/data/{tag}.npz', **cams)
with open(f'/workspace/data/{tag}_obs.json', 'w') as f:
    json.dump(obs, f)
print('step', obs['native_step'], 'episode', obs['episode'], 'dev_remaining', obs['development_remaining'])
