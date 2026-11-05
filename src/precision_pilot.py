"""Claude Code research in an offline container; native simulators stay outside.

The official RAX model relay/container bridge are reused. Benchmark adapters,
robot RPC and experiment accounting are this project's adaptation, not RAX reproduction.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import socketserver
import subprocess
import sys
import threading
import time
import traceback

import imageio.v2 as imageio
import numpy as np
from precision_envs import TASKS, NativeVisualEnv, plain, composite, png64, encoded_arrays

ROOT=Path(__file__).resolve().parents[1]
from runtime_config import RAX, MODEL, EFFORT, IMAGE, CLAUDE


def now(): return datetime.now(timezone.utc).isoformat()


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(plain(data),indent=2));p.replace(path)


class Experiment:
    def __init__(self,task,out,development=20,heldout=3,env_factory=None,resume=False,workspace=None):
        self.task,self.out,self.development,self.heldout=task,out,development,heldout
        self.env=(env_factory or NativeVisualEnv)(task);self.phase='development';self.index=0
        self.development_seed_base=int(os.environ.get('ECA_DEVELOPMENT_SEED_BASE','863000'))
        self.evaluation_seed_base=int(os.environ.get('ECA_EVALUATION_SEED_BASE','973000'))
        self.active=False;self.closed=[];self.writer=None;self.work=workspace if workspace is not None else out/'workspace'
        self.work.mkdir(exist_ok=resume);self.inbox=queue.Queue();self.freeze=False

    def reset(self):
        if self.active:self.finish('agent_reset')
        limit=self.development if self.phase=='development' else self.heldout
        if self.index>=limit:raise RuntimeError('No initial states left in this phase')
        self.index+=1
        self.seed=(getattr(self,'development_seed_base',863000) if self.phase=='development'
                   else getattr(self,'evaluation_seed_base',973000))+self.index
        self.env.reset(self.seed)
        self.episode=self.out/'episodes'/f'{self.phase}_{self.index:02d}'
        self.episode.mkdir(parents=True)
        self.active=True;self.actions=[]
        self.writer=imageio.get_writer(str(self.episode/'rgbd_all_views.mp4'),fps=self.env.hz,
            codec='libx264',macro_block_size=2,output_params=['-threads','2','-movflags','+faststart'])
        write(self.episode/'initial.json',dict(seed=self.seed,phase=self.phase,observation=self.env.public,
            hz=self.env.hz,native_horizon=self.env.spec['horizon']))
        self.record();return self.observe()

    def observe(self):
        return {**self.env.public,'phase':self.phase,'episode':self.index,
                'active':self.active,'development_remaining':max(0,self.development-self.index) if self.phase=='development' else 0}

    def archive_code(self):
        archive=self.out/'code_versions';archive.mkdir(exist_ok=True)
        hashes={}
        for p in self.work.rglob('*'):
            if p.is_symlink() or not p.is_file() or p.stat().st_size>10*1024*1024:continue
            if p.suffix not in ['.py','.json','.npy','.npz','.txt','.md']:continue
            data=p.read_bytes();digest=hashlib.sha256(data).hexdigest();hashes[str(p.relative_to(self.work))]=digest
            dest=archive/(digest+p.suffix)
            if not dest.exists():dest.write_bytes(data)
        return hashes

    def record(self):
        self.writer.append_data(composite(self.env.views,self.env.steps,self.env.hz))
        arrays={}
        for name,(rgb,depth) in self.env.views.items():
            arrays[name+'_rgb']=rgb;arrays[name+'_depth_m']=depth
        d=self.episode/'observations';d.mkdir(exist_ok=True)
        np.savez_compressed(d/f'{self.env.steps:06d}.npz',**arrays)
        with (self.episode/'observations.jsonl').open('a') as f:
            f.write(json.dumps(self.observe())+'\n')
        with (self.episode/'native_operator_audit.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=self.env.steps,audit=self.env.native_audit()))+'\n')
        self.status()

    def status(self,**extra):
        write(self.out/'status.json',dict(at=now(),task=self.task,phase=self.phase,
            episode=self.index,steps=self.env.steps if self.env.env else 0,active=self.active,
            current_success=self.env.success if self.env.env else False,closed=self.closed,**extra))

    def finish(self,reason):
        if not self.active:return
        self.active=False
        self.writer.close();self.writer=None
        np.save(self.episode/'actions.npy',np.asarray(self.actions))
        outcome=dict(task=self.task,phase=self.phase,episode=self.index,seed=self.seed,
            steps=self.env.steps,native_success=self.env.success,native_return=self.env.native_return,
            native_terminal=self.env.done,reason=reason,at=now())
        write(self.episode/'outcome.json',outcome);self.closed.append(outcome);self.status()

    def inspect(self):
        # All views, with explicit column and depth semantics; raw arrays available by RPC.
        img=composite(self.env.views,self.env.steps,self.env.hz)
        return dict(observation=self.observe(),images=[png64(img)],
            image_description='Each column is one calibrated camera; upper RGB, lower metric-depth display (0–2 m, near white). Full resolution RGB-D is available through robot.cameras().')

    def dispatch(self,request):
        kind=request['kind']
        if kind=='inspect':return self.inspect()
        if kind=='observe':return dict(observation=self.observe())
        if kind=='cameras':return encoded_arrays(self.env.views)
        if kind=='reset':
            if self.phase!='development':raise RuntimeError('No resets during frozen evaluation')
            return self.reset()
        if kind=='finish':
            if self.phase!='development':raise RuntimeError('No manual finish during frozen evaluation')
            self.finish('agent_finish');return {'finished':True}
        if kind=='step':
            if not self.active:raise RuntimeError('Episode ended; spend a counted development reset if available')
            repeat=int(request.get('repeat',1))
            if not 1<=repeat<=1000:raise ValueError('repeat must be 1..1000')
            hashes=self.archive_code() if self.phase=='development' else {}
            rewards=[]
            for _ in range(repeat):
                before=self.env.native_return
                self.env.step(request['action']);self.actions.append(request['action'])
                rewards.append(self.env.native_return-before);self.record()
                if self.env.done:self.finish('native_success' if self.env.success else 'native_horizon');break
            with (self.out/'robot_calls.jsonl').open('a') as f:
                f.write(json.dumps(dict(at=now(),phase=self.phase,episode=self.index,request=request,
                    end_step=self.env.steps,code_hashes=hashes))+'\n')
            return dict(observation=self.observe(),terminal=self.env.done,native_return=self.env.native_return,
                native_success=self.env.success,rewards=rewards)
        raise ValueError('Unknown public robot operation')

    def serve(self,proc,seconds,container):
        began=time.monotonic()
        while proc.poll() is None:
            self.status(run_status='running',operator_pid=os.getpid(),agent_container=container,
                remaining_wall_seconds=max(0,seconds-(time.monotonic()-began)))
            if time.monotonic()-began>seconds:
                subprocess.run(['docker','stop','--time','10',container],capture_output=True,timeout=40)
                return 'wall_budget'
            try:request,event,row=self.inbox.get(timeout=.2)
            except queue.Empty:continue
            try:reply=self.dispatch(request)
            except Exception:reply={'error':traceback.format_exc(limit=6)}
            # Queue items are lists. Update the one consumed by the handler.
            row[0]=reply
            event.set()
        return 'process_exit'


def create_servers(exp,relay,container):
    # Both servers use small JSON-line protocols; no simulator module is mounted in Docker.
    def invoke(request):
        event=threading.Event();slot=[None];exp.inbox.put([request,event,slot])
        if not event.wait(1800):raise TimeoutError('Robot call timed out')
        return slot[0]

    class RobotHandler(socketserver.StreamRequestHandler):
        def handle(self):
            try:result=invoke(json.loads(self.rfile.readline(16*1024*1024)))
            except Exception:result={'error':traceback.format_exc(limit=4)}
            self.wfile.write((json.dumps(result)+'\n').encode())

    class MCPHandler(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                request=json.loads(line)
                if 'id' not in request:continue
                method=request['method'];args=request.get('params',{})
                result={}
                if method=='initialize':
                    result={'protocolVersion':args.get('protocolVersion','2024-11-05'),'capabilities':{'tools':{}},
                            'serverInfo':{'name':'native_precision_robot','version':'1.0'}}
                elif method=='tools/list':
                    result={'tools':[dict(name='observe',description='Inspect calibrated multi-camera RGB-D and robot proprioception; no object GT and no physics advance.',inputSchema={'type':'object','properties':{},'additionalProperties':False}),
                        dict(name='run_python_file',description='Run a Python file from your workspace inside this offline container. Use robot_api to observe, read RGB-D, act and spend counted development resets. Full output and code versions are recorded.',
                            inputSchema={'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False})]}
                elif method=='tools/call':
                    started=time.time();execution={}
                    if args['name']=='run_python_file':
                        rel=Path(args['arguments']['path'])
                        if rel.is_absolute() or '..' in rel.parts or rel.suffix!='.py':
                            execution={'error':'Run a relative Python path within the workspace'}
                        else:
                            try:
                                done=subprocess.run(['docker','exec','--workdir','/workspace',container,'python3','-u','/workspace/'+str(rel)],
                                    capture_output=True,text=True,timeout=1800)
                                execution=dict(returncode=done.returncode,stdout=done.stdout,stderr=done.stderr)
                            except subprocess.TimeoutExpired:
                                execution={'error':'Program exceeded 1800 second wall budget'}
                    snapshot=invoke({'kind':'inspect'});images=snapshot.pop('images',[])
                    record=dict(at=now(),request=request,execution=execution,observation=snapshot)
                    with (exp.out/'mcp.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    execution={k:v[-20000:] if isinstance(v,str) else v for k,v in execution.items()}
                    blocks=[dict(type='text',text=json.dumps({**execution,**snapshot}))]
                    blocks.extend(dict(type='image',mimeType='image/png',data=x) for x in images)
                    result={'content':blocks,'isError':bool(execution.get('error') or execution.get('returncode',0))}
                elif method not in ['ping','notifications/initialized']:
                    self.wfile.write((json.dumps(dict(jsonrpc='2.0',id=request['id'],error={'code':-32601,'message':'Unsupported method'}))+'\n').encode());continue
                self.wfile.write((json.dumps(dict(jsonrpc='2.0',id=request['id'],result=result))+'\n').encode());self.wfile.flush()

    class Server(socketserver.ThreadingUnixStreamServer):daemon_threads=True
    servers=[]
    for name,handler in [('robot.sock',RobotHandler),('mcp.sock',MCPHandler)]:
        server=Server(str(relay/name),handler);threading.Thread(target=server.serve_forever,daemon=True).start();servers.append(server)
    return servers


def docker_base(out,work,home,relay,name,model_access=True):
    bridge=RAX/'harness/codex_cli/container_bridge.py'
    cmd=['docker','run','--rm','--pull=never','--name',name,'--interactive','--network','none','--read-only',
         '--user',f'{os.getuid()}:{os.getgid()}','--cap-drop=ALL','--security-opt=no-new-privileges','--init',
         '--pids-limit','512','--memory','8g','--memory-swap','8g','--cpus','4','--ulimit','core=0',
         '--tmpfs','/tmp:rw,nosuid,nodev,size=1024m,mode=1777','--workdir','/workspace']
    mounts=[(work,'/workspace',False),(ROOT/'src/robot_api.py','/opt/robot/robot_api.py',True)]
    if model_access:
        mounts += [(home,'/claude-home',False),(relay,'/run/agent-relay',True),
                   (bridge,'/opt/agent/container_bridge.py',True),
                   (CLAUDE,'/opt/claude/claude',True)]
    else:
        # Frozen programs receive only the robot socket. There is no model relay,
        # agent binary or agent conversation in the evaluation container.
        mounts += [(relay/'robot.sock','/run/agent-relay/robot.sock',True)]
    for src,dst,ro in mounts:cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
    for val in ['HOME=/claude-home','CLAUDE_CONFIG_DIR=/claude-home','PYTHONPATH=/opt/robot',
                'ROBOT_RPC_SOCKET=/run/agent-relay/robot.sock',
                'AGENT_BINARY=/opt/claude/claude','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
                'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1','CLAUDE_CODE_AUTO_COMPACT_WINDOW=200000',
                'CLAUDE_CODE_MAX_OUTPUT_TOKENS='+os.environ.get('ECA_MAX_RESPONSE_TOKENS','128000'),'ENABLE_TOOL_SEARCH=false','DISABLE_AUTOUPDATER=1',
                'ENABLE_PROMPT_CACHING_1H=1','CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL=1h']:
        cmd+=['--env',val]
    return cmd


def run(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    exp=Experiment(args.task,out,development=args.development,heldout=args.heldout);exp.reset()
    # Qualification is an independent, labelled simulator execution, with zero paid model calls.
    if args.qualify:
        a=np.zeros_like(exp.env.low);a[-1]=-1
        for _ in range(10):exp.dispatch(dict(kind='step',action=a.tolist()))
        checks={k:dict(rgb_std=float(rgb.std()),valid_depth_fraction=float((np.isfinite(d)&(d>0)).mean()),
            central_depth_fraction=float((np.isfinite(d[d.shape[0]//4:3*d.shape[0]//4,d.shape[1]//4:3*d.shape[1]//4])
                & (d[d.shape[0]//4:3*d.shape[0]//4,d.shape[1]//4:3*d.shape[1]//4]>0)).mean()))
                for k,(rgb,d) in exp.env.views.items()}
        # Zero depth on sky/background is expected; require a visible, depth-covered workspace.
        assert all(v['rgb_std']>5 and v['valid_depth_fraction']>.5 and v['central_depth_fraction']>.9
                   for v in checks.values()),checks
        exp.finish('operator_sensor_action_qualification')
        write(out/'qualification.json',dict(passed=True,paid_model_calls=0,cameras=checks,steps=10,
            public_observation=exp.observe(),action_docs=exp.env.action_docs()))
        exp.env.close();return
    write(exp.work/'TASK.json',dict(**TASKS[args.task],budget={'development_episodes':args.development,'frozen_holdout_episodes':args.heldout,
        'agent_wall_seconds':args.wall_seconds,'model_guard_usd':1000},
        action_low=exp.env.low,action_high=exp.env.high,action_docs=exp.env.action_docs(),
        observation=exp.observe(),protocol='Native default horizon, stop on first native success. Physics pauses during LLM computation. No object GT, dense reward, expert actions or learned policy.'))
    (exp.work/'API.txt').write_text('''Use import robot_api as robot. robot.observe() returns robot state, camera calibration, sparse success and episode counters. robot.cameras() returns a flat dict of <camera>_rgb uint8 HWC and <camera>_depth_m float32 HW arrays. Depth is metres. Camera matrices are in observation['cameras']; use their documented coordinate convention. robot.step(action,repeat=1) advances counted native control steps and returns observation, terminal and native_success. robot.reset() spends another development episode; no resets during heldout. robot.finish() closes this episode without claiming success. You can write arbitrary Python feedback, image processing, IK/planning, memory, diagnostics and recovery logic. NumPy/SciPy/OpenCV are available. Use the actual coding tools and robot MCP; do not access simulator internals or recreate uncounted native environments. The simulator is not installed in your offline container. Save chronological hypotheses, code changes, expected and actual outcomes to memory/PROGRESS.md. Write controller.py with main() as your frozen complete-episode entrypoint; it may call robot.observe/cameras/step and helper files, and must handle its own feedback without an online LLM. Do not call reset or finish inside main(). Native success is the criterion; do not fabricate successful states. Read TASK.json for native controller action scaling and quaternion convention. You have 20 development episodes, then your whole workspace is frozen and main() evaluated on three unseen seeds. No subagents. Your final answer does not determine success; only native rollouts do.\n''')
    camera_docs=('Pixel centres use column+0.5,row+0.5 with the intrinsic matrix. Depth is metres; nonpositive depth is invalid. '
        'ManiSkill extrinsic_cv maps world into CV camera coordinates; invert it for unprojection. '
        'Robosuite cam2world_cv maps CV camera coordinates into world coordinates; RGB and depth have already been vertically flipped together. '
        'ManiSkill metric depth is quantized at 1 mm. These are sensor conventions, not object state.\n')
    (exp.work/'CAMERA_CONVENTIONS.txt').write_text(camera_docs)
    (exp.work/'PUBLIC_MODEL.txt').write_text(json.dumps(plain(exp.env.action_docs()),indent=2)+'\n'+camera_docs+
        '\nNo full scene model.xml or object state is supplied.\n')
    reference=ROOT/'references'/args.task/'demonstration'
    if reference.exists():
        shutil.copytree(reference,exp.work/'demonstration')
        with (exp.work/'PUBLIC_MODEL.txt').open('a') as f:
            f.write('Official visual-only task references are in demonstration/. Read the PNGs; no expert actions are supplied.\n')
    (exp.work/'memory').mkdir()
    relay=Path('/tmp')/('actionabs-precision-'+hashlib.sha256(str(out).encode()).hexdigest()[:12])
    relay.mkdir(mode=0o700)
    write(out/'relay_location.json',dict(path=str(relay),reason='Short native UNIX socket path'))
    home=out/'claude-home';home.mkdir()
    write(home/'settings.json',dict(permissions={'defaultMode':'bypassPermissions'},skipDangerousModePermissionPrompt=True,
        autoUpdates=False,env={'ANTHROPIC_BASE_URL':'http://127.0.0.1:17777','ANTHROPIC_AUTH_TOKEN':'local-experiment',
        'ANTHROPIC_MODEL':MODEL,'ANTHROPIC_DEFAULT_OPUS_MODEL':MODEL,'ANTHROPIC_DEFAULT_SONNET_MODEL':MODEL,
        'ANTHROPIC_DEFAULT_HAIKU_MODEL':MODEL,'API_TIMEOUT_MS':'10830000','MCP_TOOL_TIMEOUT':'1800000',
        'MCP_TIMEOUT':'60000','ENABLE_TOOL_SEARCH':'false','DISABLE_AUTOUPDATER':'1'}))
    write(home/'.claude.json',dict(hasCompletedOnboarding=True,bypassPermissionsModeAccepted=True,
        projects={'/workspace':{'hasTrustDialogAccepted':True,'hasCompletedProjectOnboarding':True}}))
    write(home/'mcp.json',{'mcpServers':{'robot':{'command':'python3','args':['/opt/agent/container_bridge.py','mcp']}}})
    if args.smoke:
        name='actionabs-precision-smoke-'+args.task+'-'+str(os.getpid())
        servers=create_servers(exp,relay,name)
        (exp.work/'smoke.py').write_text('import robot_api as r\nimport json\ns=json.load(open("TASK.json"))\na=[0.0]*len(s["action_low"])\na[-1]=-1.0\nprint(r.observe())\nprint({k:v.shape for k,v in r.cameras().items()})\nprint(r.step(a,repeat=10))\n')
        cmd=docker_base(out,exp.work,home,relay,name)+['--entrypoint','python3',IMAGE,
            '/opt/agent/container_bridge.py','mcp']
        messages=[dict(jsonrpc='2.0',id=1,method='initialize',params={'protocolVersion':'2024-11-05'}),
            dict(jsonrpc='2.0',id=2,method='tools/list'),
            dict(jsonrpc='2.0',id=3,method='tools/call',params={'name':'run_python_file','arguments':{'path':'smoke.py'}})]
        try:
            with (out/'smoke_mcp.jsonl').open('w') as log,(out/'smoke_stderr.log').open('w') as err:
                proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=log,stderr=err)
                proc.stdin.write(('\n'.join(map(json.dumps,messages))+'\n').encode());proc.stdin.close()
                why=exp.serve(proc,180,name)
            returned=[json.loads(x) for x in (out/'smoke_mcp.jsonl').read_text().splitlines()]
            assert proc.returncode==0 and exp.env.steps==10 and len(returned)==3, (proc.returncode,exp.env.steps,len(returned))
            assert not returned[-1]['result'].get('isError'),returned[-1]
            exp.finish('operator_container_mcp_qualification')
            write(out/'qualification.json',dict(passed=True,paid_model_calls=0,mcp_handshake=True,
                native_steps=10,offline_container=True,read_only_simulator=True))
            exp.status(run_status='smoke_complete')
        finally:
            for server in servers:server.shutdown();server.server_close()
            exp.finish('qualification_cleanup');exp.env.close()
            subprocess.run(['docker','rm','--force',name],capture_output=True)
        return
    dummy=out/'proxy_dummy_token';dummy.write_text('local-experiment');dummy.chmod(0o600)
    sys.path.insert(0,str(RAX))
    from harness.claude_cli.messages_relay import MessagesRelay,Credential
    model=MessagesRelay(relay/'model.sock',Credential('custom',key_file=dummy,base_url=args.model_base_url),
        base_url=args.model_base_url,models=[MODEL],client_write_timeout=10800,slots=2,allow_private_http=True,
        usage_log=out/'model_usage.jsonl');model.start()
    name='actionabs-precision-'+args.task+'-'+str(os.getpid());servers=create_servers(exp,relay,name)
    prompt=('Read TASK.json and API.txt, inspect robot observations, and actually solve this manipulation task. '
            'Use all necessary coding, numerical/image processing and feedback control. Inspect the calibrated wrist and scene RGB-D. '
            'Iteratively test hypotheses in counted development episodes; maintain memory/PROGRESS.md with observations, failures and changes. '
            'Spend the development budget usefully, and do not stop after only proposing code or one failed attempt. '
            'Leave your best controller.py defining main() and all required helper/parameter files. '
            'Test that full controller on real development rollouts before finishing. Do not spawn subagents.')
    command=docker_base(out,exp.work,home,relay,name)+['--entrypoint','python3',IMAGE,
        '/opt/agent/container_bridge.py','-p',prompt,'--model',MODEL,'--effort',EFFORT,
        '--permission-mode','bypassPermissions','--mcp-config','/claude-home/mcp.json','--strict-mcp-config',
        '--disallowedTools','WebFetch','WebSearch','Agent','Task','--max-turns',str(args.max_turns),'--output-format','stream-json','--verbose']
    write(out/'launch.json',dict(at=now(),operator_pid=os.getpid(),container=name,argv=command,
        model=MODEL,effort='max',native_task=TASKS[args.task],harness='Native Claude Code; RAX Messages/container bridge; custom simulator adapter'))
    try:
        with (out/'claude_code.jsonl').open('w') as log,(out/'claude_stderr.log').open('w') as err:
            proc=subprocess.Popen(command,stdout=log,stderr=err,start_new_session=True)
            reason=exp.serve(proc,args.wall_seconds,name)
        exp.finish('agent_session_ended')
        write(out/'development_result.json',dict(returncode=proc.poll(),reason=reason,episodes=exp.closed,at=now()))
        if proc.returncode!=0 or not (exp.work/'controller.py').is_file():
            exp.status(run_status='agent_incomplete',returncode=proc.returncode);return
        if any(p.is_symlink() for p in exp.work.rglob('*')):raise RuntimeError('Frozen workspace contains symlinks')
        frozen=out/'frozen_workspace';shutil.copytree(exp.work,frozen)
        hashes={str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.rglob('*') if p.is_file()}
        write(out/'frozen_manifest.json',hashes)
        exp.phase='heldout';exp.index=0
        for index in range(args.heldout):
            exp.reset();eval_name=name+'-eval-'+str(index)
            evalcmd=docker_base(out,frozen,home,relay,eval_name,model_access=False)
            # Freeze code and parameter mounts during independent execution.
            evalcmd=[x.replace(f'src={frozen},dst=/workspace',f'src={frozen},dst=/workspace,readonly') for x in evalcmd]
            evalcmd+=['--entrypoint','python3',IMAGE,'-B','-u','-c','from controller import main; main()']
            with (out/f'heldout_{index+1}.log').open('w') as log:
                ev=subprocess.Popen(evalcmd,stdout=log,stderr=log);why=exp.serve(ev,1800,eval_name)
            exp.finish('frozen_program_returned')
            write(out/f'heldout_{index+1}_execution.json',dict(returncode=ev.poll(),reason=why,native_success=exp.env.success))
        changed=[n for n,h in hashes.items() if hashlib.sha256((frozen/n).read_bytes()).hexdigest()!=h]
        write(out/'freeze_audit.json',dict(changed_files=changed))
        exp.status(run_status='completed')
    finally:
        exp.finish('operator_cleanup')
        for server in servers:server.shutdown();server.server_close()
        model.close();exp.env.close()
        subprocess.run(['docker','rm','--force',name],capture_output=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=TASKS);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--qualify',action='store_true');p.add_argument('--wall-seconds',type=int,default=28800)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--model-base-url',default='http://127.0.0.1:33672')
    args=p.parse_args()
    try:run(args)
    except Exception:
        if args.out.exists():write(args.out/'operator_failure.json',dict(at=now(),traceback=traceback.format_exc()))
        raise
