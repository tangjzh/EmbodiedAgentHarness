"""Run a finite set of independent EFE agent trials; persist heartbeat and per-task status."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eca.summary import harbor_trials


def now(): return datetime.now(timezone.utc).isoformat()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tasks',nargs='+',required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--base-url',default='https://api.anthropic.com')
    p.add_argument('--model',default='claude-opus-5-5')
    p.add_argument('--allow-model-host',action='store_true')
    p.add_argument('--credential-file',type=Path)
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'batch.json').exists():raise ValueError('Use a new batch output directory')
    env=dict(os.environ)
    if a.credential_file:
        from dotenv import dotenv_values
        credentials=dotenv_values(a.credential_file)
        key=credentials.get('ANTHROPIC_API_KEY') or credentials.get('TOKEN360_API_KEY')
        if not key:raise ValueError('No API key in credential file')
        env['ANTHROPIC_API_KEY']=key
    env['ANTHROPIC_BASE_URL']=a.base_url
    env['RCB_LIMITS_ENFORCE']='1'
    state={'started_at':now(),'heartbeat_at':now(),'state':'running','model':a.model,
           'agent':'claude-code','efe_mode':'standard','workers':a.workers,
           'no_cross_task_transfer':True,'official_leaderboard_submission':False,
           'tasks':{t:{'state':'queued'} for t in a.tasks}}
    lock=threading.Lock()
    def save():
        state['heartbeat_at']=now()
        tmp=out/'batch.tmp';tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
        tmp.replace(out/'batch.json')
    def launch(task):
        destination=out/task
        cmd=[sys.executable,'-B','-m','eca','develop',task,'--out',str(destination),
             '--model',a.model,'--base-url',a.base_url]
        if a.allow_model_host:cmd.append('--allow-model-host')
        with lock:
            state['tasks'][task]={'state':'launching','started_at':now(),'output':str(destination)};save()
        with (out/(task+'.launcher.log')).open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            with lock:state['tasks'][task].update(state='running',pid=proc.pid);save()
            code=proc.wait()
        trials=harbor_trials(destination) if destination.exists() else []
        with lock:
            state['tasks'][task].update(state='finished' if code==0 else 'runner_error',
                returncode=code,finished_at=now(),trials=trials);save()
        return code
    with lock:save()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(launch,t) for t in a.tasks]
        while not all(f.done() for f in futures):
            time.sleep(15)
            with lock:save()
        errors=[]
        for f in futures:
            try:f.result()
            except Exception as e:errors.append(str(e))
    with lock:
        state.update(state='finished',finished_at=now(),orchestration_errors=errors);save()


if __name__=='__main__':main()
