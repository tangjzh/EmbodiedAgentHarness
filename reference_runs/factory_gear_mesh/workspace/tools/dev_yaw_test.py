"""Development: run controller with teeth deliberately rotated ~4 deg off to exercise the yaw sweep."""
import sys
sys.path.insert(0, '/workspace')
import controller
controller.TEST_YAW_OFFSET = 0.07
sys.argv = ['x']
exec(open('/workspace/tools/dev_episode.py').read().replace('import controller\n', ''))
