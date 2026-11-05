import sys, json
import numpy as np
sys.path.insert(0, '/workspace')
import robot_api as robot
from motion import *
from geom import *

c = np.array([0.0529, -0.2341]); ax = np.array([-0.1773, 0.9842]); L = 0.1089; r = 0.0227
yaw = np.arctan2(ax[1], ax[0])
gxy = c - (0.5 * L + 0.01) * ax
p_t = np.array([gxy[0], gxy[1], 0.10]); R_t = topdown_R(yaw)
obs = robot.observe()
log = []
for i in range(14):
    p, R = tcp_pose(obs)
    a, ep, er = servo_action(p, R, p_t, R_t, grip=1.0, kp=1.0, kr=1.0)
    res = robot.step(a.tolist())
    if i == 0:
        print('step keys', list(res.keys()))
    obs = res['observation'] if 'observation' in res else robot.observe()
    p2, R2 = tcp_pose(obs)
    log.append(dict(i=i, p=p.tolist(), a=a.tolist(), p_next=p2.tolist(), ep=ep.tolist(), er=er.tolist(),
                    dp=(p2 - p).tolist(), drot=rotvec(R2 @ R.T).tolist(), qpos=obs['robot']['agent']['qpos'][0],
                    step=obs['native_step']))
    print(i, 'err %.4f rot %.4f' % (np.linalg.norm(ep), np.linalg.norm(er)), 'cmd', np.round(a[:3] * 0.1, 4), 'moved', np.round(p2 - p, 4),
          'rotcmd', np.round(a[3:6] * 0.1, 4), 'rotmoved', np.round(rotvec(R2 @ R.T), 4))
json.dump(log, open('/workspace/data/ep1_a_log.json', 'w'))
cams = robot.cameras()
np.savez_compressed('/workspace/data/ep1_pregrasp.npz', **cams)
json.dump(obs, open('/workspace/data/ep1_pregrasp_obs.json', 'w'))
print('final tcp', tcp_pose(obs)[0], 'native_step', obs['native_step'])
