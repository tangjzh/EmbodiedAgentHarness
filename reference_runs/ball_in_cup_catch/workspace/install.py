"""Archive current controller.py once as controller_v1.py, then install the version named in INSTALL_FROM."""
import os
import shutil

INSTALL_FROM = "controller_v3.py"
if not os.path.exists("controller_v1.py"):
    shutil.copy("controller.py", "controller_v1.py")
    print("archived v1")
shutil.copy(INSTALL_FROM, "controller.py")
print("installed", INSTALL_FROM)
