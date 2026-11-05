import os, sys
root = os.path.dirname(os.path.abspath(__file__))
for dp, dn, fn in os.walk(root):
    depth = dp[len(root):].count(os.sep)
    if depth > 2:
        continue
    for f in sorted(fn):
        p = os.path.join(dp, f)
        print(os.path.relpath(p, root), os.path.getsize(p))
print("python", sys.version)
for mod in ["numpy", "scipy", "cv2"]:
    try:
        m = __import__(mod)
        print(mod, getattr(m, "__version__", "?"))
    except Exception as e:
        print(mod, "ERR", e)
