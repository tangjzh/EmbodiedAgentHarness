import sys, json
import numpy as np
sys.path.insert(0, '/workspace')
import robot_api as robot
from motion import *
from geom import *

obs = robot.observe()
cmds = [[0, 0, 0, 0, 0, 0.5, 1], [0, 0, 0, 0, 0, 0.5, 1], [0, 0, 0, 0.5, 0, 0, 1], [0, 0, 0, 0.5, 0, 0, 1],
        [0, 0, 0, 0, 0.5, 0, 1], [0, 0, 0, 0, 0.5, 0, 1]]
log = []
for a in cmds:
    p, R = tcp_pose(obs)
    res = robot.step(a)
    obs = res['observation']
    p2, R2 = tcp_pose(obs)
    w = rotvec(R2 @ R.T)
    wb = rotvec(R.T @ R2)
    print('cmd', a[3:6], 'world rot', np.round(w, 4), 'body rot', np.round(wb, 4), 'dp', np.round(p2 - p, 4))
    log.append(dict(a=a, p=p.tolist(), R=R.tolist(), p2=p2.tolist(), R2=R2.tolist(), qpos=obs['robot']['agent']['qpos'][0]))
json.dump(log, open('/workspace/data/ep1_rot_log.json', 'w'))
json.dump(obs, open('/workspace/data/ep1_rot_obs.json', 'w'))
