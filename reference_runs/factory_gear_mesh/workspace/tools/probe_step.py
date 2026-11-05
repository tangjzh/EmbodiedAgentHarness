import robot_api as robot, json
r = robot.step([0,0,0,0,0,0], repeat=1)
print(json.dumps({k: v for k, v in r.items() if k != 'observation'}, default=str)[:2000])
print(list(r.keys()))
print(list(r.get('observation', {}).keys()))
