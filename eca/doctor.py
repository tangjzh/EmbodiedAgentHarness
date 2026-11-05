"""Read-only environment checks. Does not start a model or simulator."""
import hashlib
import importlib.util
from importlib.metadata import version,PackageNotFoundError
import os
from pathlib import Path
import shutil
from .catalog import ROOT
def inspect(item):
    checks=[]
    def check(name,ok,detail):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    if item['backend']=='state':
        for name,module in [('dm-control','dm_control'),('gymnasium','gymnasium'),('mujoco','mujoco'),('mcp','mcp')]:
            check(name,importlib.util.find_spec(module) is not None,'Install requirements/state.txt in Python 3.11')
        check('bubblewrap',shutil.which('bwrap'),'State programs run in an isolated numerical runtime')
    elif item['backend']=='capx':
        root=Path(os.environ.get('ECA_CAPX_ROOT',ROOT/'external/cap-x'))
        check('CaP-X',root.is_dir(),str(root))
        check('Docker',shutil.which('docker'),'Agent process is offline')
        check('perception services','ECA_SAM3_URL' in os.environ and 'ECA_GRASPNET_URL' in os.environ and 'ECA_IK_URL' in os.environ,
            'Set endpoints and launch pinned upstream SAM3, Contact-GraspNet and PyRoki services')
    else:
        check('Docker',shutil.which('docker'),'Required for the offline agent and frozen executor')
        root=Path(os.environ.get('ECA_RAX_ROOT',ROOT/'external/rax-robodojo'))
        if item['backend']=='rax_dexjoco':root=Path(os.environ.get('ECA_DEXJOCO_RAX_ROOT',ROOT/'external/rax-dexjoco'))
        check('RAX source',root.is_dir(),str(root))
        if item['group'] in ['factory','robodojo']:check('RTX rendering node',shutil.which('nvidia-smi'),'Factory/RoboDojo camera qualification requires the RTX node used by the original setup')
    binary=os.environ.get('ECA_CLAUDE_BIN') or shutil.which('claude')
    check('Claude Code binary',binary and Path(binary).is_file(),'Historical binary 2.1.304; configure ECA_CLAUDE_BIN. Closed-source binary is not bundled.')
    if binary and Path(binary).is_file():
        got=hashlib.sha256(Path(binary).read_bytes()).hexdigest()
        check('historical CLI SHA256',got=='ea0f135784631619a7a409d07322ecec964f9cc4fd377508ed1ba3f141f27912',got)
    return dict(task=item['id'],passed=all(c['passed'] for c in checks),checks=checks,paid_model_calls=0)
