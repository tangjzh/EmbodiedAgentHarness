import sys, json
import numpy as np
sys.path.insert(0, '/workspace')
import robot_api as robot
tag = open('/workspace/data/snap_tag.txt').read().strip()
obs = robot.observe()
cams = robot.cameras()
np.savez_compressed(f'/workspace/data/{tag}.npz', **cams)
json.dump(obs, open(f'/workspace/data/{tag}_obs.json', 'w'))
print('saved', tag, obs['native_step'])
