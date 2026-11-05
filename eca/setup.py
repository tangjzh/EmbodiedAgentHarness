"""Print setup commands; clone/install only when explicitly executed."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
from .catalog import ROOT
GROUPS={'state':[],'maniskill':['rax-robodojo','maniskill'],'dexmimicgen':['rax-robodojo','dexmimicgen'],
 'factory':['rax-robodojo','robodojo','isaaclab'],'dlo':['rax-robodojo','dlo-lab'],
 'robodojo':['rax-robodojo','robodojo','isaaclab'],'dexjoco':['rax-dexjoco','dexjoco'],
 'capx_libero':['rax-robodojo','cap-x','libero-pro'],'capx_robosuite':['rax-robodojo','cap-x','capx-robosuite']}
def setup(args):
    if args.group not in GROUPS:raise ValueError('Unknown environment group')
    upstreams=json.loads((ROOT/'provenance/upstreams.json').read_text());commands=[]
    for key in GROUPS[args.group]:
        spec=upstreams[key];path=ROOT/'external'/spec.get('path',key)
        if not (path/'.git').exists():
            if path.exists() and any(path.iterdir()):
                raise ValueError('Existing non-Git directory must be preserved: '+str(path))
            commands.append(['git','clone','--no-checkout',spec['url'],str(path)])
        commands.append(['git','-C',str(path),'checkout','--detach',spec['commit']])
    requirement={'state':'state','maniskill':'precision','dexmimicgen':'precision','factory':'factory','dlo':'dlo'}.get(args.group)
    venv=args.venv or ROOT/('.venvs/'+args.group)
    if requirement:commands.extend([[sys.executable,'-m','venv',str(venv)],
        [str(venv/'bin/python'),'-m','pip','install','-r',str(ROOT/'requirements'/(requirement+'.txt'))]])
    for command in commands:
        print(shlex.join(command),flush=True)
        if args.execute:subprocess.run(command,check=True,cwd=ROOT)
    if args.group!='state':print('Complete native runtime, Docker, patches and assets in docs/ENVIRONMENTS.md; cloning is not simulator qualification.')
