import numpy as np, robot_api as robot, controller
o = robot.observe(); obs = o['observation'] if 'observation' in o else o
m1, m2 = {}, {}
a1 = controller.policy(obs, 0, m1)
flat = list(obs['position']) + list(obs['velocity']) + list(obs['touch'])
a2 = controller.policy(flat, 0, m2)
print('dict ->', a1, ' flat ->', a2, ' mem', m1)
