import sys
import robot_api as robot
from controller import policy
memory={}
obs=robot.observe()
for tick in range(10000):
    action=policy(obs,tick,memory)
    result=robot.step(action)
    obs=result['observation']
    if result['terminal']: break
print({'ticks':tick+1,'native_return':result['native_return']})
