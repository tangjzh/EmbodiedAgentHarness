import robot_api as robot, numpy as np, json, sys
tag = 'cur'
o = robot.observe()
json.dump(o, open(f'/tmp/obs_{tag}.json','w'), default=lambda x: np.asarray(x).tolist())
np.savez(f'/tmp/cams_{tag}.npz', **robot.cameras())
print(list(o.keys()))
