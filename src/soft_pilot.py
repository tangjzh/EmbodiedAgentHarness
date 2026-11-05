"""Real Claude Code development on native deformable simulators.

Terminal continuous metrics and evaluation-only cloth resets are kept distinct
from observations and robot actions. No simulator is mounted into the agent.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np
from precision_envs import plain, composite, png64
from precision_pilot import Experiment, create_servers, docker_base, write, now, ROOT, RAX, MODEL
from runtime_config import EFFORT, IMAGE
from soft_envs import TASKS, make_env


class SoftExperiment(Experiment):
    def __init__(self, task, out, development=20, heldout=3, resume=False):
        super().__init__(task, out, development, heldout, env_factory=make_env, resume=resume)
        self.env.record_callback = self.record_if_active

    def record_if_active(self):
        if self.active and self.writer is not None:
            self.record()

    def finish(self, reason):
        if not self.active: return
        self.active = False
        # GarmentLab's native criterion resets the garment to calculate a
        # reference area. Close the action video BEFORE this scoring operation.
        self.writer.close()
        self.writer = None
        native_terminal_before_scoring = self.env.done
        self.env.metric_video_dir = self.episode
        metric = self.env.finalize_metric()
        write(self.episode/'actions.json', self.actions)
        if self.task != 'garment_fling':
            np.save(self.episode/'actions.npy', np.asarray(self.actions))
        outcome = dict(task=self.task, phase=self.phase, episode=self.index, seed=self.seed,
            steps=self.env.steps, physical_steps=getattr(self.env, 'physics_steps',
                self.env.steps*getattr(self.env, 'per_action_physics_steps',20)),
            native_success=self.env.success, native_return=self.env.native_return,
            native_metric=metric, native_terminal=native_terminal_before_scoring,
            native_horizon_exhausted=self.env.steps>=self.env.spec['horizon'], reason=reason, at=now())
        write(self.episode/'outcome.json', outcome)
        self.closed.append(outcome)
        self.status()

    def dispatch(self, request):
        if request['kind'] == 'command':
            if not self.active: raise RuntimeError('Episode ended')
            if not hasattr(self.env, 'execute_command'): raise ValueError('Native commands are unavailable for this task')
            start = self.env.steps
            hashes = self.archive_code() if self.phase == 'development' else {}
            self.actions.append(request)
            self.env.execute_command(request['operation'], **request.get('arguments', {}))
            if self.env.done: self.finish('native_horizon')
            with (self.out/'robot_calls.jsonl').open('a') as f:
                f.write(json.dumps(dict(at=now(), phase=self.phase, episode=self.index,
                    request=request, start_step=start, end_step=self.env.steps, code_hashes=hashes))+'\n')
            return dict(observation=self.observe(), terminal=self.env.done, native_success=self.env.success,
                final_native_metric=self.env.final_metric if not self.active else None)
        if request['kind'] != 'step':
            reply = super().dispatch(request)
            if request['kind'] == 'finish':
                reply['final_native_metric'] = self.env.final_metric
            return reply
        if not self.active: raise RuntimeError('Episode ended; use a counted development reset if available')
        repeat = int(request.get('repeat', 1))
        if not 1 <= repeat <= 1000: raise ValueError('repeat must be 1..1000')
        hashes = self.archive_code() if self.phase == 'development' else {}
        start = self.env.steps
        for _ in range(repeat):
            self.actions.append(request['action'])
            self.env.step(request['action'])
            if not self.env.auto_record: self.record()
            if self.env.done:
                self.finish('native_horizon_or_terminal')
                break
        with (self.out/'robot_calls.jsonl').open('a') as f:
            f.write(json.dumps(dict(at=now(), phase=self.phase, episode=self.index, request=request,
                start_step=start, end_step=self.env.steps, code_hashes=hashes))+'\n')
        # No dense reward or hidden geometry reaches the agent during a rollout.
        return dict(observation=self.observe(), terminal=self.env.done, native_success=self.env.success,
            final_native_metric=self.env.final_metric if not self.active else None)


def prepare(exp, args):
    write(exp.work/'TASK.json', dict(**TASKS[args.task],
        budget=dict(development_episodes=args.development, frozen_holdout_episodes=args.heldout,
            agent_wall_seconds=args.wall_seconds, model_guard_usd=1000),
        action_low=exp.env.low, action_high=exp.env.high, neutral_action=exp.env.neutral_action,
        action_docs=exp.env.action_docs(), observation=exp.observe(),
        protocol='Physics pauses during LLM computation. Native score is terminal-only. No object/particle GT or expert action sequence.'))
    (exp.work/'API.txt').write_text('''Use import robot_api as robot. robot.observe() returns only robot proprioception, camera calibration and episode counters. robot.cameras() returns a flat dict of <camera>_rgb uint8 HWC and <camera>_depth_m float32 HW arrays. Camera matrices are in observation['cameras']. robot.step(action,repeat=1) executes documented native commands and returns observation/terminal/final_native_metric. robot.reset() spends a development episode, closing any active one and evaluating its final native metric. robot.finish() closes the episode and returns the native terminal metric; it does not claim success. Read TASK.json and PUBLIC_MODEL.txt before choosing commands. You may write arbitrary numerical/image processing, feedback, planning, memory and recovery code using NumPy, SciPy, OpenCV and the RGB-D cameras. No simulator or benchmark source is available inside this offline container. Keep chronological hypotheses, code changes and observed outcomes in memory/PROGRESS.md. Develop controller.py defining main() as the frozen complete-episode entrypoint: it observes and controls from sensors with helper files, without an online LLM, reset() or finish(). It must return at a terminal state. You have 20 development episodes followed by 3 runs of the frozen workspace. No subagents. Use the full useful development budget, test your full controller, and preserve failures. Final prose cannot determine the result. The native DLO-Lab metric is continuous and has NO native success threshold; never invent a success percentage for it.\n''')
    camera_docs = ('CV cameras: x right, y down, z forward. cam2world_cv maps camera to world. '
        'Depth is axial metres, not Euclidean distance. Nonpositive/nonfinite depth is invalid. '
        'Use depth_pixel_ray_offset and rgb_pixel_ray_offset from the calibrated metadata. '
        'DLO simultaneous MSAA depth uses u+0.375,v+0.875, nominal RGB u+0.5,v+0.5; Factory uses u+0.5,v+0.5.\n'
        'Frozen execution mounts the workspace read-only. Save runtime diagnostics only under /tmp.\n')
    (exp.work/'CAMERA_CONVENTIONS.txt').write_text(camera_docs)
    (exp.work/'PUBLIC_MODEL.txt').write_text(json.dumps(plain(exp.env.action_docs()), indent=2)+'\n'+camera_docs)
    (exp.work/'memory').mkdir(exist_ok=True)


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    exp = SoftExperiment(args.task, out, development=args.development, heldout=args.heldout)
    try:
        exp.reset()
    except Exception:
        exp.env.close()
        raise
    prepare(exp, args)
    if args.qualify:
        initial = exp.observe()
        a = exp.env.neutral_action.copy()
        if args.task == 'dlo_unknotting': a[2] = .2
        if args.task != 'garment_fling':
            for _ in range(10): exp.dispatch(dict(kind='step', action=a.tolist()))
        else:
            exp.dispatch(dict(kind='command', operation='wait', arguments={'ticks':10}))
        checks = {k:dict(rgb_std=float(rgb.std()),
            valid_depth_fraction=float((np.isfinite(d)&(d>0)).mean()))
            for k,(rgb,d) in exp.env.views.items()}
        assert all(v['rgb_std'] > 5 and v['valid_depth_fraction'] > .3 for v in checks.values()), checks
        exp.finish('operator_sensor_action_qualification')
        write(out/'qualification.json', dict(passed=True, paid_model_calls=0, cameras=checks,
            native_steps=exp.env.steps, before=initial, after=exp.observe(), action_docs=exp.env.action_docs(),
            final_native_metric=exp.env.final_metric))
        exp.env.close()
        return
    relay = Path('/tmp')/('actionabs-soft-'+hashlib.sha256(str(out).encode()).hexdigest()[:12])
    relay.mkdir(mode=0o700)
    write(out/'relay_location.json', dict(path=str(relay)))
    home = out/'claude-home'
    home.mkdir()
    write(home/'settings.json', dict(permissions={'defaultMode':'bypassPermissions'},
        skipDangerousModePermissionPrompt=True, autoUpdates=False,
        env=dict(ANTHROPIC_BASE_URL='http://127.0.0.1:17777', ANTHROPIC_AUTH_TOKEN='local-experiment',
            ANTHROPIC_MODEL=MODEL, ANTHROPIC_DEFAULT_OPUS_MODEL=MODEL, ANTHROPIC_DEFAULT_SONNET_MODEL=MODEL,
            ANTHROPIC_DEFAULT_HAIKU_MODEL=MODEL, API_TIMEOUT_MS='10830000', MCP_TOOL_TIMEOUT='1800000',
            MCP_TIMEOUT='60000', ENABLE_TOOL_SEARCH='false', DISABLE_AUTOUPDATER='1')))
    write(home/'.claude.json', dict(hasCompletedOnboarding=True, bypassPermissionsModeAccepted=True,
        projects={'/workspace':{'hasTrustDialogAccepted':True,'hasCompletedProjectOnboarding':True}}))
    write(home/'mcp.json', dict(mcpServers={'robot':{'command':'python3',
        'args':['/opt/agent/container_bridge.py','mcp']}}))
    if args.smoke:
        name = 'actionabs-soft-smoke-'+args.task+'-'+str(os.getpid())
        servers = create_servers(exp, relay, name)
        action_code = ('print(r.step(s["neutral_action"],repeat=10))\n' if args.task!='garment_fling'
            else 'print(r.command("wait",ticks=10))\n')
        (exp.work/'smoke.py').write_text('import robot_api as r\nimport json\ns=json.load(open("TASK.json"))\n'
            'print(r.observe())\nprint({k:v.shape for k,v in r.cameras().items()})\n'
            +action_code)
        cmd = docker_base(out,exp.work,home,relay,name)+['--entrypoint','python3',IMAGE,
            '/opt/agent/container_bridge.py','mcp']
        messages = [dict(jsonrpc='2.0',id=1,method='initialize',params={'protocolVersion':'2024-11-05'}),
            dict(jsonrpc='2.0',id=2,method='tools/list'), dict(jsonrpc='2.0',id=3,method='tools/call',
                params={'name':'run_python_file','arguments':{'path':'smoke.py'}})]
        try:
            with (out/'smoke_mcp.jsonl').open('w') as log,(out/'smoke_stderr.log').open('w') as err:
                proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=log,stderr=err)
                proc.stdin.write(('\n'.join(map(json.dumps,messages))+'\n').encode());proc.stdin.close()
                why=exp.serve(proc,600,name)
            returned=[json.loads(x) for x in (out/'smoke_mcp.jsonl').read_text().splitlines()]
            assert proc.returncode==0 and exp.env.steps>=10 and len(returned)==3,(proc.returncode,exp.env.steps,len(returned))
            assert not returned[-1]['result'].get('isError'),returned[-1]
            exp.finish('operator_container_mcp_qualification')
            write(out/'qualification.json',dict(passed=True,paid_model_calls=0,mcp_handshake=True,
                native_steps=exp.env.steps,offline_container=True,final_native_metric=exp.env.final_metric))
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
    name='actionabs-soft-'+args.task+'-'+str(os.getpid())
    servers=create_servers(exp,relay,name)
    prompt=('Read TASK.json, API.txt and PUBLIC_MODEL.txt, inspect all RGB-D cameras, and actually solve this deformable task. '
        'Use coding, visual geometry, feedback, planning and recovery. Preserve chronological observations, hypotheses, '
        'code changes and failures in memory/PROGRESS.md. Test your full controller on counted development rollouts; '
        'use the available development episodes to improve it. Read and trust the documented native action units. '
        'Leave controller.py defining main() plus required helpers as the best frozen feedback program. '
        'Do not spawn subagents or invent success thresholds. Only simulator evaluation determines results.')
    cmd=docker_base(out,exp.work,home,relay,name)+['--entrypoint','python3',IMAGE,
        '/opt/agent/container_bridge.py','-p',prompt,'--model',MODEL,'--effort',EFFORT,
        '--permission-mode','bypassPermissions','--mcp-config','/claude-home/mcp.json','--strict-mcp-config',
        '--disallowedTools','WebFetch','WebSearch','Agent','Task','--max-turns',str(args.max_turns),
        '--output-format','stream-json','--verbose']
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')}
    write(out/'launch.json',dict(at=now(),operator_pid=os.getpid(),container=name,argv=cmd,model=MODEL,
        effort='max',native_task=TASKS[args.task],operator_source_hashes=source_hashes,
        harness='Native Claude Code with RAX model/container bridge; custom native visual task adapter'))
    shutil.copytree(ROOT/'src',out/'operator_source',ignore=shutil.ignore_patterns('__pycache__'))
    try:
        with (out/'claude_code.jsonl').open('w') as log,(out/'claude_stderr.log').open('w') as err:
            proc=subprocess.Popen(cmd,stdout=log,stderr=err,start_new_session=True)
            reason=exp.serve(proc,args.wall_seconds,name)
        exp.finish('agent_session_ended')
        write(out/'development_result.json',dict(returncode=proc.poll(),reason=reason,episodes=exp.closed,at=now()))
        if proc.returncode!=0 or not (exp.work/'controller.py').is_file():
            exp.status(run_status='agent_incomplete',returncode=proc.returncode)
            return
        if any(p.is_symlink() for p in exp.work.rglob('*')):raise RuntimeError('Frozen workspace contains symlinks')
        frozen=out/'frozen_workspace';shutil.copytree(exp.work,frozen)
        hashes={str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.rglob('*') if p.is_file()}
        write(out/'frozen_manifest.json',hashes)
        exp.phase='heldout';exp.index=0
        for index in range(args.heldout):
            exp.reset();eval_name=name+'-eval-'+str(index)
            evcmd=docker_base(out,frozen,home,relay,eval_name,model_access=False)
            evcmd=[x.replace(f'src={frozen},dst=/workspace',f'src={frozen},dst=/workspace,readonly') for x in evcmd]
            evcmd+=['--entrypoint','python3',IMAGE,'-B','-u','-c','from controller import main; main()']
            with (out/f'heldout_{index+1}.log').open('w') as log:
                ev=subprocess.Popen(evcmd,stdout=log,stderr=log);why=exp.serve(ev,3600,eval_name)
            exp.finish('frozen_program_returned')
            write(out/f'heldout_{index+1}_execution.json',dict(returncode=ev.poll(),reason=why,
                native_success=exp.env.success,native_metric=exp.env.final_metric))
        changed=[n for n,h in hashes.items() if hashlib.sha256((frozen/n).read_bytes()).hexdigest()!=h]
        write(out/'freeze_audit.json',dict(changed_files=changed))
        exp.status(run_status='completed')
    finally:
        exp.finish('operator_cleanup')
        for server in servers:server.shutdown();server.server_close()
        model.close();exp.env.close()
        subprocess.run(['docker','rm','--force',name],capture_output=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=TASKS);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--qualify',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--wall-seconds',type=int,default=28800)
    p.add_argument('--model-base-url',default='http://127.0.0.1:33674')
    args=p.parse_args()
    try:run(args)
    except Exception:
        if args.out.exists():write(args.out/'operator_failure.json',dict(at=now(),traceback=traceback.format_exc()))
        raise
