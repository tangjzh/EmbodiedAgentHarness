import robot_api as robot, numpy as np, json, sys
tag = sys.argv[1] if len(sys.argv) > 1 else 'dump'
obs = robot.observe()
cams = robot.cameras()
np.savez_compressed(f'/workspace/tools/{tag}.npz', **cams)
json.dump(obs, open(f'/workspace/tools/{tag}_obs.json', 'w'))
print({k: v.shape for k, v in cams.items()})
