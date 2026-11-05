import json
import numpy as np
import robot_api as robot
from controller import policy
assert callable(policy), 'controller.policy must be callable'
obs=robot.observe()
a=np.asarray(policy(obs,0,{}),dtype=float)
spec=json.load(open('TASK.json'))
lo=np.asarray(spec['native_action_low']);hi=np.asarray(spec['native_action_high'])
assert a.shape==lo.shape and np.all(np.isfinite(a)), 'invalid policy action shape/values'
assert np.all(a>=lo-1e-7) and np.all(a<=hi+1e-7), 'policy action outside native limits'
print('controller import and first action validated; no physics advanced')
