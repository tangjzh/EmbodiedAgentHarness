import inspect

import robot_api as robot

print([n for n in dir(robot) if not n.startswith("_")])
try:
    print(robot.docs())
except Exception as e:
    print("docs error", e)
for n in dir(robot):
    if n.startswith("_"):
        continue
    obj = getattr(robot, n)
    if callable(obj):
        try:
            print(n, inspect.signature(obj))
        except Exception:
            print(n, "(no signature)")
        if obj.__doc__:
            print("   ", obj.__doc__.strip()[:500])
