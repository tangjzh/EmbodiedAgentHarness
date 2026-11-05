import sys, json
sys.path.insert(0, '/workspace')
import robot_api as robot
if not robot.observe().get('active', False):
    robot.reset()
import controller
controller.main()
L = json.load(open(controller.LOG))
for e in L['events']: print(json.dumps(e)[:400])
for s in L['steps'][::4]: print(s['i'], s['a'][:3], s['eef'], s['succ'])
print('final', L.get('final'), 'first succ', next((s['i'] for s in L['steps'] if s['succ']), None))
