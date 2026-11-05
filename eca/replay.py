"""Run saved task-local code without a model relay or cross-task initialization."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from .catalog import ROOT
def visual(item,args,out,module):
    from precision_pilot import Experiment,create_servers,docker_base,write
    from runtime_config import IMAGE
    source=ROOT/item['reference']
    if not (source/'controller.py').is_file():raise ValueError('No complete frozen controller is archived for this task')
    out.mkdir(parents=True,exist_ok=False)
    experiment_cls=getattr(module,'SoftExperiment',Experiment)
    exp=experiment_cls(item['id'],out,development=1,heldout=item['heldout_episodes'])
    shutil.copytree(source,exp.work,dirs_exist_ok=True)
    hashes={str(p.relative_to(exp.work)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in exp.work.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    relay=Path(tempfile.mkdtemp(prefix='eca-replay-'));home=out/'unused-home'
    name='eca-replay-'+str(os.getpid());servers=create_servers(exp,relay,name)
    exp.phase='heldout';exp.index=0
    try:
        for i in range(item['heldout_episodes']):
            exp.reset();container=name+'-'+str(i)
            command=docker_base(out,exp.work,home,relay,container,model_access=False)
            command=[x.replace('src='+str(exp.work)+',dst=/workspace','src='+str(exp.work)+',dst=/workspace,readonly') for x in command]
            command+=['--entrypoint','python3',IMAGE,'-B','-u','-c','from controller import main; main()']
            with (out/('heldout_'+str(i+1)+'.log')).open('w') as log:
                proc=subprocess.Popen(command,stdout=log,stderr=log)
                reason=exp.serve(proc,1800,container)
            exp.finish('frozen_program_returned')
            write(out/('heldout_'+str(i+1)+'_execution.json'),dict(returncode=proc.returncode,reason=reason,
                native_success=exp.env.success,new_model_calls=0))
        changed=[n for n,h in hashes.items() if hashlib.sha256((exp.work/n).read_bytes()).hexdigest()!=h]
        write(out/'freeze_audit.json',dict(changed_files=changed,new_model_calls=0,reference_task=item['id']))
        if changed:raise RuntimeError('Frozen reference program changed')
        exp.status(run_status='completed',new_model_calls=0)
    finally:
        exp.finish('replay_cleanup')
        for server in servers:server.shutdown();server.server_close()
        exp.env.close()
        shutil.rmtree(relay)
