import numpy as np, json, sys
sys.path.insert(0, '/workspace')
import perception
d = dict(np.load('/tmp/cams_cur.npz'))
o=json.load(open('/tmp/obs_cur.json')); calib=o['cameras']
r = perception.estimate(d, calib, o['robot']['eef_position_world'][:2])
print(json.dumps(r, indent=1))
