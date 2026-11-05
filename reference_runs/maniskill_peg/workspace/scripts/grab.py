import sys, json, time
import numpy as np
sys.path.insert(0, '/opt/robot')
import robot_api as robot

tag = 'ep1_init'
obs = robot.observe()
cams = robot.cameras()
print({k: (v.shape, v.dtype, float(np.nanmin(v)), float(np.nanmax(v))) for k, v in cams.items()})
np.savez_compressed(f'/workspace/data/{tag}.npz', **cams)
with open(f'/workspace/data/{tag}_obs.json', 'w') as f:
    json.dump(obs, f)
print(json.dumps({k: obs[k] for k in obs if k not in ('cameras',)})[:2000])
