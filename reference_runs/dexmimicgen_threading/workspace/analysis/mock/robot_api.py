"""Kinematic mock of robot_api for offline dry runs of controller.main() (static saved images)."""
import json, os
import numpy as np
from scipy.spatial.transform import Rotation as Rot

TAG = os.environ.get('MOCK_TAG', 'grab')
_d = dict(np.load(f'/workspace/data/{TAG}.npz'))
_o = json.load(open(f'/workspace/data/{TAG}_obs.json'))
CAM_IN_HAND = np.array([[0., -1., 0., 0.05], [1., 0., 0., 0.], [0., 0., 1., -0.097], [0., 0., 0., 1.]])
BASE = [np.array([-0.56, -0.25]), np.array([-0.56, 0.25])]
S = dict(step=0)
for i in range(2):
    r = _o['robot']
    S[f'p{i}'] = np.array(r[f'robot{i}_eef_pos'])
    S[f'R{i}'] = Rot.from_quat(r[f'robot{i}_eef_quat']).as_matrix()
    S[f'g{i}'] = 0.02
    S[f'j7{i}'] = r[f'robot{i}_joint_pos'][6]
GRIP_HOLD = [float(os.environ.get('MOCK_G0', 0.0119)), float(os.environ.get('MOCK_G1', 0.0150))]
SUCC_AT = int(os.environ.get('MOCK_SUCC', 10 ** 9))


def _obs():
    o = json.loads(json.dumps(_o))
    for i in range(2):
        p, R = S[f'p{i}'], S[f'R{i}']
        r = o['robot']
        r[f'robot{i}_eef_pos'] = p.tolist()
        r[f'robot{i}_eef_quat'] = Rot.from_matrix(R).as_quat().tolist()
        r[f'robot{i}_gripper_qpos'] = [S[f'g{i}'], -S[f'g{i}']]
        jp = list(r[f'robot{i}_joint_pos']); jp[6] = S[f'j7{i}']; r[f'robot{i}_joint_pos'] = jp
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
        o['cameras'][f'robot{i}_eye_in_hand']['cam2world_cv'] = (T @ CAM_IN_HAND).tolist()
    o['native_step'] = S['step']
    o['native_success'] = S['step'] >= SUCC_AT
    o['terminal'] = o['native_success'] or S['step'] >= 1000
    return o


def observe():
    return _obs()


def cameras():
    return dict(_d)


def step(action, repeat=1):
    a = np.asarray(action, float)
    for _ in range(repeat):
        for i in range(2):
            ai = a[7 * i:7 * i + 7]
            S[f'p{i}'] = S[f'p{i}'] + 0.3 * ai[:3] * 0.05
            dR = Rot.from_rotvec(0.3 * ai[3:6] * 0.5).as_matrix()
            Rn = dR @ S[f'R{i}']
            dyaw = np.arctan2(Rn[1, 0], Rn[0, 0]) - np.arctan2(S[f'R{i}'][1, 0], S[f'R{i}'][0, 0])
            dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
            S[f'j7{i}'] -= dyaw
            S[f'R{i}'] = Rn
            S[f'g{i}'] = GRIP_HOLD[i] if ai[6] > 0 else 0.02
        S['step'] += 1
    o = _obs()
    return dict(observation=o, terminal=o['terminal'], native_success=o['native_success'])


def reset():
    raise RuntimeError('mock: no reset')


def finish():
    pass
