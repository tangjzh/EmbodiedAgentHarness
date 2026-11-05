"""Public robot client. The simulator, rewards and accounting live in another process."""
import json
import os
import socket
from pathlib import Path
import atexit
import sys

_announced_docs = False


def _call(kind, **kwargs):
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(650)
        sock.connect(os.environ.get('ROBOT_RPC_SOCKET', '/run/robot/gateway.sock'))
        stream = sock.makefile('rwb')
        stream.write((json.dumps(dict(kind=kind, **kwargs))+'\n').encode())
        stream.flush()
        reply = json.loads(stream.readline())
    if 'error' in reply:
        raise RuntimeError(reply['error'])
    return _calibrated_metadata(reply)


def observe():
    """Native benchmark observation, current reward and episode metadata; no physics advance."""
    global _announced_docs
    if not _announced_docs and Path('/workspace/PUBLIC_MODEL.txt').exists():
        print('Public sensor and controller documentation: PUBLIC_MODEL.txt. Use docs() to read the specification. It contains no episode-specific hidden state.')
        _announced_docs = True
    return _call('observe')['observation']


def step(action, repeat=1):
    """Apply one native action for repeat control ticks (1..1000), stopping at native termination."""
    return _call('step', action=[float(x) for x in action], repeat=int(repeat))


def reset():
    """Close the current development episode and spend one fresh development initial state."""
    return _call('reset')


def command(operation, **arguments):
    """Execute a native task command when supported; see PUBLIC_MODEL.txt."""
    return _call('command', operation=operation, arguments=arguments)


def finish():
    """Close this episode. Does not report success or start another episode."""
    return _call('finish')


def cameras():
    """Current RGB uint8 and metric depth float32 arrays; no physics advance."""
    import base64
    import numpy as np
    values = _call('cameras')
    return {k:np.frombuffer(base64.b64decode(v['data']),dtype=v['dtype']).reshape(v['shape']).copy()
            for k,v in values.items()}


def docs():
    """Public sensor/action meanings and static model documentation, not simulator state."""
    return Path(os.environ.get('ROBOT_TASK_DOCS','/workspace/PUBLIC_MODEL.txt')).read_text()


def task():
    """Read the current public task specification, including episode budgets."""
    return json.loads(Path(os.environ.get('ROBOT_TASK_SPEC','/workspace/TASK.json')).read_text())


def _calibrated_metadata(reply):
    """Apply an operator-issued sensor convention notice, without altering data."""
    path = Path('/workspace/SENSOR_CALIBRATION.json')
    if not path.exists():
        return reply
    observation = reply.get('observation')
    if observation is None and 'cameras' in reply:
        observation = reply  # reset() returns the public observation directly.
    if not isinstance(observation, dict): return reply
    calibration = json.loads(path.read_text())
    observation['sensor_calibration_revision'] = calibration['revision']
    for camera in observation.get('cameras', {}).values():
        camera['depth_pixel_ray_offset'] = calibration['depth_pixel_ray_offset']
        camera['rgb_pixel_ray_offset'] = calibration['rgb_pixel_ray_offset']
        camera['pixel_centres'] = calibration['description']
    return reply


def _print_operator_notice():
    path = Path('/workspace/OPERATOR_NOTICE.txt')
    if path.exists():
        print('\nOPERATOR SENSOR / EXECUTION NOTICE:\n'+path.read_text(), file=sys.stderr, flush=True)


atexit.register(_print_operator_notice)
