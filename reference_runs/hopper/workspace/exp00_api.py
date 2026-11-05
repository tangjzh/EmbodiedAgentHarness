import inspect, os
import robot_api as robot

print("robot_api attributes:", [a for a in dir(robot) if not a.startswith('_')])
for name in ['observe', 'step', 'reset', 'finish', 'cameras']:
    f = getattr(robot, name, None)
    if f is None:
        print(name, 'MISSING'); continue
    try:
        print(name, inspect.signature(f))
    except Exception as e:
        print(name, 'sig err', e)
    doc = getattr(f, '__doc__', None)
    if doc:
        print('   doc:', doc.strip()[:400])
o = robot.observe()
print(type(o))
if isinstance(o, dict):
    for k, v in o.items():
        s = repr(v)
        print(' ', k, ':', s[:200])
print('workspace files:', sorted(os.listdir('.')))
# --- stub factory (continuation session has no file-create tool; stubs are then filled via Edit) ---
import pathlib
_names = ['hopctl3.py', 'hopctl4.py', 'hopctl5.py', 'analyze.py', 'runctl.py'] + ['exp%02d.py' % i for i in range(3, 21)]
for _n in _names:
    _p = pathlib.Path(_n)
    if not _p.exists():
        _p.write_text('# stub\n')
        print('created stub', _n)
