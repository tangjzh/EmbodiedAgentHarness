"""Delegate native RoboDojo/DexJoCo to the pinned upstream RAX workflow."""
import json
import os
from pathlib import Path
import subprocess
import sys
from .catalog import ROOT
def root_for(item):
    name='rax-dexjoco' if item['backend']=='rax_dexjoco' else 'rax-robodojo'
    key='ECA_DEXJOCO_RAX_ROOT' if name=='rax-dexjoco' else 'ECA_RAX_ROOT'
    return Path(os.environ.get(key,ROOT/'external'/name)).resolve()
def run(item,args,out,base):
    repo=root_for(item)
    if not (repo/'scripts/configure_auto_research.py').is_file():raise RuntimeError('Run eca setup for '+item['group']+'; RAX source is missing')
    python=Path(os.environ.get('ECA_RAX_PYTHON',repo/'runtime/envs/host/bin/python'))
    if not python.is_file():raise RuntimeError('Prepare the pinned RAX host environment; see docs/ENVIRONMENTS.md')
    out.mkdir(parents=True,exist_ok=False)
    bench='dexjoco' if item['backend']=='rax_dexjoco' else 'robodojo'
    task='hammer_nail' if bench=='dexjoco' else item['id']
    configure=[str(python),'scripts/configure_auto_research.py','--benchmark',bench,'--task',task,
        '--sim-gpu',str(args.gpu),'--episodes',str(item['development_episodes']),
        '--agent-cli','claude','--agent-provider','custom','--agent-effort',item['effort'],
        '--external-perception','off','--agent-formal','off','--formal-episodes','3','--formal-workers','1',
        '--subagents','0','--exploration-envs','1','--observation-profile','rgbd',
        '--demonstration-context','none' if bench=='dexjoco' else 'completion_sequence','--task-rules','code-summary']
    done=subprocess.run(configure,cwd=repo,capture_output=True,text=True)
    (out/'configure_stdout.log').write_text(done.stdout);(out/'configure_stderr.log').write_text(done.stderr)
    done.check_returncode()
    paths=[Path(v) for v in done.stdout.splitlines() if v.startswith('/') and v.endswith('operator-input.json')]
    if len(paths)!=1:raise RuntimeError('Expected one newly created RAX configuration')
    config=paths[0];settings=json.loads(config.read_text())
    if settings.get('adaptation_source') or settings.get('agent_formal'):raise RuntimeError('Fresh code-only protocol violated')
    if settings.get('write_handoff'):raise RuntimeError('Cross-task handoff must remain disabled')
    (out/'session.json').write_text(json.dumps(dict(configuration=str(config),native_root=settings['root'],
        fresh_workspace=True,task=item['id']),indent=2))
    env=dict(os.environ,PYTHONPATH=str(repo)+os.pathsep+str(ROOT/'src'),
        CLAUDE_CODE_MAX_OUTPUT_TOKENS='128000')
    if args.command=='qualify':
        subprocess.run([str(python),str(ROOT/'src/rax_replay.py'),'--config',str(config),
            '--qualify','--out',str(out/'qualification')],cwd=repo,env=env,check=True);return
    if args.command=='replay':
        subprocess.run([str(python),str(ROOT/'src/rax_replay.py'),'--config',str(config),
            '--reference',str(ROOT/item['reference']),'--task',item['id'],'--out',str(out/'replay')],cwd=repo,env=env,check=True);return
    key=out/'loopback-placeholder.key';key.write_text('local-experiment\n');key.chmod(0o600)
    launcher=('from harness.claude_cli.messages_relay import PROVIDERS; '
        'import os; model=os.environ["ECA_MODEL"]; '
        'PROVIDERS["custom"].update(model=model,auditor=model,fast=model); '
        'from harness.codex_cli.auto_research import main; main()')
    command=[str(python),'-c',launcher,'--config',str(config),'--non-interactive',
        '--claude-key-file',str(key),'--claude-base-url',base,'--claude-model',args.model]
    with (out/'operator.log').open('w') as log:subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=log,check=True)
