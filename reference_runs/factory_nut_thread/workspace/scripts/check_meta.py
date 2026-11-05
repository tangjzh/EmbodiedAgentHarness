"""Check whether observe() carries pixel-ray offset metadata (claimed by a notice seen in a tool result)."""
import json

import robot_api as robot

obs = robot.observe()


def find(d, path=''):
    if isinstance(d, dict):
        for k, v in d.items():
            p = f'{path}.{k}' if path else k
            if 'offset' in k or 'pixel' in k or 'ray' in k:
                print(p, '=', json.dumps(v)[:200])
            find(v, p)


find(obs)
print('top-level keys', list(obs.keys()))
print('camera keys', {k: list(v.keys()) for k, v in obs['cameras'].items()})
print('episode', obs.get('episode'), 'active', obs.get('active'), 'dev remaining', obs.get('development_remaining'), 'step', obs.get('native_step'))
