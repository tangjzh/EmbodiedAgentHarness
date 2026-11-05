"""Exploratory dynamic-task case studies. Claude Code owns the online coding loop.

New benchmark adapters use the CaP-X-style MCP/code runtime and RAX's
explore -> program -> verify workflow. This is an adaptation, not a RAX paper reproduction.
"""
import argparse
import base64
import faulthandler
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import socketserver
import sys
import subprocess
import threading
import time
import traceback

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from run_checks import agent_outcome

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
TASKS = {
    'ball_in_cup_catch': dict(benchmark='dm_control', domain='ball_in_cup', task='catch',
        goal='Swing the tethered ball into the cup and keep it caught. Reward is native binary catch occupancy.',
        dynamics='The tethered ball must gain momentum to rise into the cup; static positioning is insufficient.'),
    'finger_spin': dict(benchmark='dm_control', domain='finger', task='spin',
        goal='Use the two-actuator finger to sustain the native rewarded spinner rotation direction, above its 15 rad/s threshold.',
        dynamics='Native reward requires sustained angular velocity, not a final pose.'),
    'hopper': dict(benchmark='Gymnasium MuJoCo', env_id='Hopper-v5',
        goal='Hop forward while staying upright. Maximize the native episode return; report distance and survival. This environment has no native binary task-success predicate.',
        dynamics='Forward hopping requires dynamic balance, takeoff and landing.'),
    'acrobot_swingup': dict(benchmark='dm_control', domain='acrobot', task='swingup',
        goal='Swing the underactuated double pendulum upright and keep its tip near the native target; maximize native reward.',
        dynamics='Only the elbow is actuated; raising the passive shoulder from the hanging state requires inertial coupling.'),
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def safe(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [safe(v) for v in value]
    return value


def write(path, value):
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(safe(value), indent=2)); temp.replace(path)


class Environment:
    def __init__(self, task, output, development=20, heldout=3):
        self.task, self.spec, self.out = task, TASKS[task], output
        self.development, self.heldout = development, heldout
        self.index = 0; self.phase = 'development'; self.active = False
        self.closed = []; self.physics_records = []; self.observation_records = []
        self.env = None; self.writer = None; self.renderer = None

    def reset(self):
        if self.active: self.close_episode('agent_reset')
        limit = self.development if self.phase == 'development' else self.heldout
        if self.index >= limit: raise RuntimeError('This phase has no unused initial states')
        if self.env is not None:
            if self.renderer is not None: self.renderer.close(); self.renderer = None
            self.env.close()
            if self.spec['benchmark']=='dm_control':self.env.physics.free()
        self.index += 1
        seed = (741000 if self.phase == 'development' else 951000)+self.index
        if self.spec['benchmark'] == 'dm_control':
            from dm_control import suite
            self.env = suite.load(self.spec['domain'], self.spec['task'], task_kwargs={'random':seed})
            ts = self.env.reset(); self.obs = ts.observation
            self.low, self.high = self.env.action_spec().minimum, self.env.action_spec().maximum
            self.hz = 1/self.env.control_timestep()
        else:
            import gymnasium as gym
            self.env = gym.make(self.spec['env_id'], render_mode='rgb_array', width=480, height=480)
            self.obs, _ = self.env.reset(seed=seed)
            self.low, self.high = self.env.action_space.low, self.env.action_space.high
            self.hz = 1/self.env.unwrapped.dt
        self.episode = self.out/'episodes'/f'{self.phase}_{self.index:02d}'
        self.episode.mkdir(parents=True)
        self.steps = 0; self.total = 0.; self.last_reward = 0.; self.active = True
        self.done = False; self.rewarded_ticks = 0; self.max_reward_streak = 0; self.streak = 0
        self.physics_records = []; self.observation_records = []; self.actions = []
        self.seed = seed
        self.writer = imageio.get_writer(str(self.episode/'rgb_depth.mp4'), fps=min(self.hz,50),
            codec='libx264', quality=7, macro_block_size=2, ffmpeg_params=['-threads','1'])
        write(self.episode/'initial.json',dict(seed=seed, phase=self.phase, task=self.task,
              native_observation=self.obs, action_min=self.low, action_max=self.high, control_hz=self.hz))
        self.record_frame()
        self.status()
        return self.observe()

    def physics(self):
        if self.spec['benchmark'] == 'dm_control':
            return self.env.physics.model, self.env.physics.data
        return self.env.unwrapped.model, self.env.unwrapped.data

    def render(self):
        if self.spec['benchmark'] == 'dm_control':
            p = self.env.physics
            return p.render(480,480,camera_id=0), p.render(480,480,camera_id=0,depth=True)
        import mujoco
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.env.unwrapped.model, height=480, width=480)
        self.renderer.update_scene(self.env.unwrapped.data, camera='track')
        self.renderer.disable_depth_rendering(); rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering(); depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return rgb,depth

    def record_frame(self):
        rgb, depth = self.render()
        # Fixed metric 0..5m map, same scale in all videos. Original depth saved at inspections.
        t = np.clip(np.nan_to_num(depth,nan=0.,posinf=5.,neginf=0.)/5.,0,1)
        colors = np.stack([255*t, 255*np.sqrt(t), 255*(1-t)],axis=-1).astype(np.uint8)
        colors[~np.isfinite(depth)] = 0
        canvas = Image.fromarray(np.concatenate([rgb,colors],axis=1))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0,0,960,32), fill='black')
        draw.text((8,7), f'{self.task} | {self.phase} {self.index} | tick {self.steps} | RGB', fill='white')
        draw.text((490,7),'Metric depth: blue=0m, yellow=5m+; invalid=black', fill='white')
        self.writer.append_data(np.asarray(canvas))
        self.rgb, self.depth = rgb, depth
        _, data = self.physics()
        self.physics_records.append(np.concatenate([np.asarray(data.qpos).copy(),np.asarray(data.qvel).copy()]))
        if getattr(self,'on_progress',None):self.on_progress()

    def observe(self):
        return safe(self.obs)

    def step(self, action, repeat=1):
        if not self.active or self.done: raise RuntimeError('Episode ended; call reset() to spend a new development episode')
        action = np.asarray(action,dtype=float)
        if action.shape != np.asarray(self.low).shape or not np.all(np.isfinite(action)):
            raise ValueError(f'Finite action shape {np.asarray(self.low).shape} required')
        if np.any(action < self.low-1e-7) or np.any(action > self.high+1e-7):
            raise ValueError('Action outside documented native bounds; clip explicitly')
        if not 1 <= repeat <= 1000: raise ValueError('repeat must be 1..1000')
        rewards = []
        for _ in range(repeat):
            if self.spec['benchmark'] == 'dm_control':
                ts = self.env.step(action); self.obs = ts.observation
                r = float(ts.reward or 0); self.done = bool(ts.last())
            else:
                self.obs,r,terminated,truncated,info = self.env.step(action)
                r = float(r); self.done = bool(terminated or truncated)
            self.steps += 1; self.total += r; self.last_reward = r
            self.actions.append(action.copy()); rewards.append(r)
            self.observation_records.append(dict(step=self.steps, observation=safe(self.obs), reward=r, done=self.done))
            rewarded = r > .5 if self.spec['benchmark'] == 'dm_control' else False
            self.rewarded_ticks += int(rewarded); self.streak = self.streak+1 if rewarded else 0
            self.max_reward_streak = max(self.max_reward_streak,self.streak)
            self.record_frame()
            if self.done: self.close_episode('native_termination'); break
        self.status()
        return dict(observation=self.observe(), rewards=rewards, native_return=self.total,
                    step=self.steps, terminal=self.done, phase=self.phase,
                    episode=self.index, development_remaining=max(0,self.development-self.index))

    def close_episode(self, reason):
        if not self.active: return
        self.active = False
        if self.writer is not None: self.writer.close(); self.writer = None
        np.savez_compressed(self.episode/'trajectory.npz', actions=np.asarray(self.actions),
                            qpos_qvel=np.asarray(self.physics_records))
        with (self.episode/'native_observations.jsonl').open('w') as f:
            for row in self.observation_records: f.write(json.dumps(row)+'\n')
        result = dict(task=self.task, phase=self.phase, episode=self.index, seed=self.seed,
            native_return=self.total, steps=self.steps, native_terminated=self.done,
            reason=reason, rewarded_ticks=self.rewarded_ticks, max_reward_streak=self.max_reward_streak,
            time=stamp(), video=str((self.episode/'rgb_depth.mp4').relative_to(self.out)))
        if self.spec['benchmark'] == 'dm_control':
            result['native_mean_reward'] = self.total/max(1,self.steps)
        else:
            _, data = self.physics(); result.update(final_x=float(data.qpos[0]), final_height=float(data.qpos[1]))
        write(self.episode/'outcome.json',result); self.closed.append(result); self.status()

    def status(self, **extra):
        write(self.out/'status.json', dict(updated_at=stamp(),task=self.task,phase=self.phase,
            current_episode=self.index, current_steps=getattr(self,'steps',0),
            current_return=getattr(self,'total',0), active=self.active,
            closed_episodes=self.closed,
            replayed_prefix_steps=getattr(self,'replayed_prefix_steps',0),
            new_native_steps=sum(e['steps'] for e in self.closed)
                +(getattr(self,'steps',0) if self.active else 0)-getattr(self,'replayed_prefix_steps',0),
            **extra))

    def inspect(self):
        if not self.active and self.env is None: self.reset()
        tag = f'{self.phase}_{self.index:02d}_{self.steps:04d}_{time.time_ns()}'
        folder = self.out/'observations'; folder.mkdir(exist_ok=True)
        rgb, depth = self.render()
        Image.fromarray(rgb).save(folder/(tag+'.jpg'),quality=95)
        np.save(folder/(tag+'.depth.npy'),depth)
        out = io.BytesIO(); Image.fromarray(rgb).save(out,format='JPEG',quality=90)
        t = np.clip(np.nan_to_num(depth,nan=0.,posinf=5.,neginf=0.)/5.,0,1)
        d = np.stack([255*t,255*np.sqrt(t),255*(1-t)],axis=-1).astype(np.uint8)
        d[~np.isfinite(depth)] = 0
        depth_out = io.BytesIO(); Image.fromarray(d).save(depth_out,format='JPEG',quality=90)
        return dict(observation=self.observe(), step=self.steps, terminal=self.done,
            native_return=self.total, phase=self.phase, episode=self.index,
            action_min=safe(self.low), action_max=safe(self.high), control_hz=self.hz,
            closed_episodes=self.closed,
            image_order=['RGB','metric depth blue=0m yellow=5m+ black=invalid'],
            images=[base64.b64encode(out.getvalue()).decode(),base64.b64encode(depth_out.getvalue()).decode()],
            observation_artifacts=[str((folder/(tag+'.jpg')).relative_to(self.out)),str((folder/(tag+'.depth.npy')).relative_to(self.out))])


def command(work, config, prompt):
    return [str(__import__('runtime_config').CLAUDE),'--bare','--restricted','-p',prompt,
        '--model',__import__('runtime_config').MODEL,'--effort',__import__('runtime_config').EFFORT,'--tools','Read,Edit,Write',
        '--allowedTools','Read,Edit,Write,mcp__robot__observe,mcp__robot__run_python_file',
        '--permission-mode','dontAsk','--permission-prompts','none','--setting-sources','',
        '--strict-mcp-config','--mcp-config',str(config),'--max-turns','200',
        '--max-budget-usd','250','--output-format','stream-json','--verbose','--no-session-persistence']


def restore_continuation(engine, source, work):
    """Rebuild saved episode prefixes, checking every available state and reward.

    This is a new CLI context, not a native Claude session resume. Replay is an
    audited reconstruction and must never be counted as another trial or new data.
    """
    source=Path(source).resolve()
    status=json.loads((source/'status.json').read_text())
    if status['task']!=engine.task or status['phase']!='development':
        raise ValueError('Continuation requires the same native task and development phase')
    prior_path=source/'development_result.json'
    prior=json.loads(prior_path.read_text()) if prior_path.exists() else {}
    prior_audit=prior.get('continuation') or (json.loads((source/'continuation_audit.json').read_text())
        if (source/'continuation_audit.json').exists() else {})
    replayed=0;max_error=0.;trace_hashes={};zero_action_starts=[]
    for index in range(1,status['current_episode']+1):
        folder=source/'episodes'/f'development_{index:02d}'
        initial=json.loads((folder/'initial.json').read_text())
        if engine.index!=index:engine.reset()
        if initial['seed']!=engine.seed:raise ValueError('Native replay seed mismatch')
        # Also verify the original public observation on zero-action starts whose
        # process died before it could flush a physical-state trace.
        if json.dumps(safe(engine.observe()),sort_keys=True)!=json.dumps(initial['native_observation'],sort_keys=True):
            raise ValueError('Native replay initial observation mismatch')
        trace=folder/'trajectory.npz'
        if not trace.exists():
            if index!=status['current_episode'] or status['current_steps']!=0:
                raise ValueError('Missing saved physical actions; refusing approximate continuation')
            zero_action_starts.append(dict(episode=index,seed=engine.seed,
                verification='same native seed and exact public initial observation; original state snapshot absent'))
            continue
        saved=np.load(trace);actions=saved['actions'];states=saved['qpos_qvel']
        observations=[json.loads(l) for l in (folder/'native_observations.jsonl').read_text().splitlines()]
        if len(states)!=len(actions)+1 or len(observations)!=len(actions):
            raise ValueError('Incomplete saved action/state trace')
        trace_hashes[str(trace.relative_to(source))]=hashlib.sha256(trace.read_bytes()).hexdigest()
        np.testing.assert_allclose(engine.physics_records[0],states[0],atol=1e-11,rtol=0)
        for i,action in enumerate(actions):
            actual=engine.step(action)
            error=float(np.max(np.abs(engine.physics_records[-1]-states[i+1])))
            max_error=max(max_error,error)
            np.testing.assert_allclose(engine.physics_records[-1],states[i+1],atol=1e-11,rtol=0)
            np.testing.assert_allclose(actual['rewards'][-1],observations[i]['reward'],atol=1e-11,rtol=0)
        replayed+=len(actions)
    if engine.done:raise ValueError('Continuation requires an unfinished current episode')
    old_work=source/'workspace'
    if any(p.is_symlink() for p in old_work.rglob('*')):
        raise ValueError('Refusing symlinks in carried workspace')
    shutil.copytree(old_work,work,dirs_exist_ok=True,ignore=shutil.ignore_patterns(
        '.git','.claude','__pycache__','TASK.json','API.txt','CONTINUATION.json'))
    feedback=work/'PRIOR_TOOL_FEEDBACK.jsonl'
    previous_feedback=old_work/'PRIOR_TOOL_FEEDBACK.jsonl'
    feedback.write_text((previous_feedback.read_text() if previous_feedback.exists() else '')
        +(source/'mcp.jsonl').read_text())
    cli=[json.loads(l) for l in (source/'claude_code.jsonl').read_text().splitlines()]
    terminal=next((r for r in reversed(cli) if r.get('type')=='result'),{})
    if 'wall_seconds' in prior:
        previous_wall=prior['wall_seconds']
    else:
        launch=json.loads((source/'launch.json').read_text())
        previous_wall=(datetime.fromisoformat(status['completed_at'])
            -datetime.fromisoformat(launch['started_at'])).total_seconds()
    previous_turns=terminal.get('num_turns')
    turn_estimate=previous_turns is None
    if turn_estimate:
        previous_turns=len({r.get('message',{}).get('id',r.get('uuid')) for r in cli if r.get('type')=='assistant'})
    report=dict(source_case=str(source),source_trace_hashes=trace_hashes,
        reconstruction='same native seed and complete action-prefix replay',
        cli_context='fresh session with carried workspace, memory and delivered tool feedback',
        replayed_prefix_steps=replayed,replay_max_state_error=max_error,
        zero_action_starts=zero_action_starts,previous_turn_count_is_estimate=turn_estimate,
        previous_agent_wall_seconds=previous_wall+prior_audit.get('previous_agent_wall_seconds',0),
        previous_agent_turns=previous_turns+prior_audit.get('previous_agent_turns',0),
        counts_as_new_trial=False,verified=True,verified_at=stamp())
    write(engine.out/'continuation_audit.json',report)
    write(work/'CONTINUATION.json',report)
    engine.replayed_prefix_steps=replayed;engine.status()
    return report


def carry_workspace(source, work):
    source=Path(source).resolve();old=source/'workspace'
    if any(p.is_symlink() for p in old.rglob('*')):raise ValueError('Refusing workspace symlinks')
    shutil.copytree(old,work,dirs_exist_ok=True,ignore=shutil.ignore_patterns(
        '.git','.claude','__pycache__','TASK.json','API.txt','CONTINUATION.json'))
    prior=old/'PRIOR_TOOL_FEEDBACK.jsonl';calls=source/'mcp.jsonl'
    (work/'PRIOR_TOOL_FEEDBACK.jsonl').write_text((prior.read_text() if prior.exists() else '')
        +(calls.read_text() if calls.exists() else ''))
    return {str(p.relative_to(work)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in work.rglob('*.py')}


def fresh_episode_resume(source, work, out):
    source=Path(source).resolve();s=json.loads((source/'status.json').read_text())
    launch=json.loads((source/'launch.json').read_text())
    inherited=json.loads((source/'continuation_audit.json').read_text()) if (source/'continuation_audit.json').exists() else {}
    result=json.loads((source/'development_result.json').read_text()) if (source/'development_result.json').exists() else {}
    wall=result.get('wall_seconds',(datetime.fromisoformat(s['completed_at'])
        -datetime.fromisoformat(launch['started_at'])).total_seconds())
    events=[json.loads(l) for l in (source/'claude_code.jsonl').read_text().splitlines() if l.strip()]
    terminal=next((r for r in reversed(events) if r.get('type')=='result'),{})
    turns=terminal.get('num_turns',len({r.get('message',{}).get('id',r.get('uuid'))
        for r in events if r.get('type')=='assistant'}))
    carry_workspace(source,work)
    report=dict(source_case=str(source),fresh_episode_after_interruption=True,
        interrupted_episode=s['current_episode'],next_episode=s['current_episode']+1,
        previous_native_steps=sum(e['steps'] for e in s['closed_episodes'])
            +(s['current_steps'] if s['active'] else 0),
        previous_agent_wall_seconds=wall+inherited.get('previous_agent_wall_seconds',0),
        previous_agent_turns=turns+inherited.get('previous_agent_turns',0),
        previous_turn_count_is_estimate='num_turns' not in terminal,
        replayed_prefix_steps=0,claims_exact_physical_continuation=False,
        cli_context='new session with original workspace, memory and delivered feedback',at=stamp())
    write(out/'continuation_audit.json',report);write(work/'CONTINUATION.json',report)
    return report


def run(task, out, gpu, development=20, heldout=3, smoke=False,
        model_base_url='http://127.0.0.1:33622', cohort='summary_v3',render_backend='osmesa',
        continue_from=None,checkpoint_from=None,resume_workspace=None):
    if sum(x is not None for x in [continue_from,checkpoint_from,resume_workspace])>1:
        raise ValueError('Select one recovery or checkpoint mode')
    os.environ.update(MUJOCO_GL=render_backend,PYOPENGL_PLATFORM=render_backend,
                      MUJOCO_EGL_DEVICE_ID=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    out.mkdir(parents=True,exist_ok=False)
    stack_file=(out/'operator_stacks.log').open('w')
    faulthandler.register(signal.SIGUSR1,file=stack_file,all_threads=True)
    work = out/'workspace'; work.mkdir()
    engine = Environment(task,out,development,heldout)
    if resume_workspace:
        prior=json.loads((Path(resume_workspace)/'status.json').read_text())
        if prior['task']!=task or prior['phase']!='development':raise ValueError('Wrong recovery source')
        engine.index=prior['current_episode']
    engine.reset()
    # This is a sensor check, with no physics advance and no paid model call.
    rgb_a,depth_a=engine.render();rgb_a=rgb_a.copy();depth_a=depth_a.copy()
    rgb_b,depth_b=engine.render()
    visual_check=dict(backend=render_backend,rgb_std=float(rgb_a.std()),
        valid_depth_fraction=float(np.isfinite(depth_a).mean()),depth_std=float(np.nanstd(depth_a)),
        same_state_rgb_mae=float(np.abs(rgb_a.astype(float)-rgb_b.astype(float)).mean()),
        same_state_depth_max_error=float(np.nanmax(np.abs(depth_a-depth_b))))
    visual_check['passed']=(visual_check['rgb_std']>5 and visual_check['valid_depth_fraction']>.999
        and visual_check['depth_std']>1e-5 and visual_check['same_state_rgb_mae']<.01
        and visual_check['same_state_depth_max_error']<1e-4)
    write(out/'sensor_qualification.json',visual_check)
    if not visual_check['passed']:
        engine.close_episode('invalid_rendering')
        engine.status(run_status='infrastructure_interrupted',reason='Sensor qualification failed; no model started')
        if engine.renderer is not None:engine.renderer.close()
        engine.env.close()
        if engine.spec['benchmark']=='dm_control':engine.env.physics.free()
        raise RuntimeError('Sensor qualification failed; no paid agent started')
    write(work/'TASK.json',dict(**TASKS[task], budget=dict(development_episodes=development,heldout_episodes=heldout,
        agent_wall_seconds=7200,agent_turns=200),native_action_low=engine.low,native_action_high=engine.high,
        initial_public_observation=engine.observe(),control_hz=engine.hz))
    api = '''Use import robot_api as robot in Python files executed by robot.run_python_file.
robot.observe() returns the benchmark's public native observation.
robot.cameras() returns {'rgb': uint8 HWC, 'depth_m': float32 HW} for CPU image analysis.
robot.step(action,repeat=1) applies native bounded actuator commands and returns observations/rewards/terminal.
You can run a feedback loop at every native control tick in your code, without an LLM call per tick.
robot.reset() ends the current development episode and consumes another; reset is NEVER free.
robot.finish() ends the current episode. Every physics tick and initial state is accounted.
NumPy, SciPy, OpenCV, image processing, file-based memory and arbitrary controller code are available.
Your files run as separate processes; persist code/parameters in the workspace between calls.
Inspect the returned RGB and native sensor values. Native rewards may be used during development.
Do not access simulator internals, other sessions, host files, credentials or internet.
There is no supplied trained policy or task-specific action primitive.
Write controller.py defining policy(observation, step, memory) -> action. memory is a persistent dict
within one rollout. This function must use only the observation argument and workspace code/parameters;
no robot.step/reset/observe calls inside policy. Controller receives observations at every native tick.
The operator will freeze it and evaluate on three unseen initial states, with no live agent or reset.
'''
    (work/'API.txt').write_text(api)
    references=ROOT/'references'/task
    if references.exists():
        for source in references.iterdir():
            target=work/source.name
            if source.is_dir():shutil.copytree(source,target)
            else:shutil.copyfile(source,target)
        (work/'API.txt').write_text(api+'\nRead PUBLIC_MODEL.txt and model.xml for native sensor meanings and public static model parameters before designing control.\n')
    (work/'controller.py').write_text('# Implement policy(observation, step, memory) -> native action.\n')
    continuation=(restore_continuation(engine,continue_from,work) if continue_from else
        fresh_episode_resume(resume_workspace,work,out) if resume_workspace else {})
    checkpoint={}
    if checkpoint_from:
        hashes=carry_workspace(checkpoint_from,work)
        checkpoint=dict(source_case=str(Path(checkpoint_from).resolve()),kind='interrupted_agent_checkpoint',
            agent_finished_normally=False,new_model_calls=0,source_code_hashes=hashes,
            evaluation='frozen program on previously unused heldout starts',at=stamp())
        write(out/'checkpoint_provenance.json',checkpoint)
    remaining_wall=7200-continuation.get('previous_agent_wall_seconds',0)
    remaining_turns=200-continuation.get('previous_agent_turns',0)
    if remaining_wall<=0 or remaining_turns<=0:
        raise ValueError('The interrupted case has exhausted its original agent budget')
    if continuation:
        task_spec=json.loads((work/'TASK.json').read_text())
        task_spec['budget'].update(remaining_agent_wall_seconds=remaining_wall,
            remaining_agent_turns=remaining_turns,carried_development_steps=engine.steps,
            continuation_is_new_trial=False)
        task_spec['current_public_observation']=engine.observe()
        write(work/'TASK.json',task_spec)
    inbox = queue.Queue()
    socket_dir = Path('/tmp')/('embodied-pilot-'+hashlib.sha256(str(out).encode()).hexdigest()[:12])
    socket_dir.mkdir(mode=0o700,exist_ok=True)
    socket_path = socket_dir/'gateway.sock'
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                payload=json.loads(self.rfile.readline(1000000)); ready=threading.Event(); item=[payload,ready,None]
                inbox.put(item)
                if not ready.wait(620): raise TimeoutError('Robot gateway timed out')
                self.wfile.write((json.dumps(item[2])+'\n').encode())
            except Exception as exc:
                self.wfile.write((json.dumps({'error':str(exc)})+'\n').encode())
    class Server(socketserver.ThreadingUnixStreamServer): daemon_threads=True
    server=Server(str(socket_path),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    config = out/'mcp.json'
    write(config,{'mcpServers':{'robot':{'type':'stdio','command':str(PYTHON),
        'args':[str(ROOT/'src/pilot_mcp.py')], 'env':{'PILOT_WORKSPACE':str(work),'PILOT_OUTPUT':str(out),'PILOT_SOCKET':str(socket_path)}}}})
    if checkpoint_from:
        args=None;env=None
    elif smoke:
        (work/'smoke.py').write_text('import robot_api as r\nprint(r.observe())\nprint(r.step([0.0]*'+str(len(engine.low))+'))\n')
        env=dict(os.environ,PILOT_WORKSPACE=str(work),PILOT_OUTPUT=str(out),PILOT_SOCKET=str(socket_path))
        args=[str(PYTHON),'-c','import pilot_mcp;print(pilot_mcp.run_code("smoke.py"))']
        env['PYTHONPATH']=str(ROOT/'src')
    else:
        prompt=('Read TASK.json and API.txt, inspect robot.observe, and solve the dynamic task through actual code execution. '
            'Explore using the full development budget, implement and test controllers, keep a chronological memory/PROGRESS.md '
            'of hypotheses and evidence, and leave your best controller.py. You have 20 development episodes; '
            'use resets as measured experiments and avoid spending all of them on an untested long loop. '
            'For each experiment briefly state the hypothesis, changed code or parameters, predicted outcome, and observed outcome. '
            'Use feedback and native rewards to optimize; do not stop at proposing code. '
            'Code versions, tool calls and trajectories are recorded for case analysis. '
            'Do not spawn subagents. Finish with a short experimental account of what worked and failed.')
        if continuation.get('fresh_episode_after_interruption'):
            prompt=('RECOVERY ON A NEW COUNTED DEVELOPMENT EPISODE: read CONTINUATION.json, '
                'memory/PROGRESS.md, PRIOR_TOOL_FEEDBACK.jsonl and the existing code. Previous simulation '
                'stopped after heartbeat loss. Its incomplete episode remains spent; no exact replay is claimed. '
                f'You are at new episode {engine.index}, tick 0. You retain {remaining_wall:.0f} wall seconds '
                f'and {remaining_turns} turns from the original budget. Continue improving the saved controller '
                'using real executions, and leave a working controller.py. Do not repeat completed probes. '+prompt)
        elif continuation:
            prompt=('CONTINUATION AFTER TRANSPORT INTERRUPTION: read CONTINUATION.json, memory/PROGRESS.md, '
                'PRIOR_TOOL_FEEDBACK.jsonl and the existing controller/experiment code first. The simulator is at '
                f'episode {engine.index} tick {engine.steps}, reconstructed from recorded native actions and starts. This is the SAME '
                'unfinished development episode; all previous ticks, resets and agent time remain spent. '
                'The code and recorded tool results are real prior experiments. The old CLI session ended due '
                'to an incorrect transport deadline while generating; it did not complete the task. You have '
                f'{remaining_wall:.0f} agent wall seconds and {remaining_turns} turns remaining. Continue from '
                'the current physical state, do not repeat probes just because this is a new CLI context. '+prompt)
        args=command(work,config,prompt)
        if continuation and '--max-turns' in args:
            args[args.index('--max-turns')+1]=str(remaining_turns)
        env={k:v for k,v in os.environ.items() if not k.startswith(('ANTHROPIC_','CLAUDE_')) and k not in
             ['TOKEN360_API_KEY','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','OPENROUTER_API_KEY','GH_TOKEN','GITHUB_TOKEN']}
        env.update(ANTHROPIC_BASE_URL=model_base_url,ANTHROPIC_API_KEY=os.environ.get('ECA_LOOPBACK_API_KEY','local-experiment'),
            ANTHROPIC_CUSTOM_HEADERS='X-Experiment-Episode: '+task+'__'+cohort,CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1',
            CLAUDE_CODE_DISABLE_AUTO_MEMORY='1',ENABLE_TOOL_SEARCH='false',DISABLE_AUTOUPDATER='1',
            CLAUDE_CONFIG_DIR=str(out/'claude-home'),CLAUDE_CODE_MAX_OUTPUT_TOKENS='128000',
            CLAUDE_STREAM_IDLE_TIMEOUT_MS='1800000',API_TIMEOUT_MS='10830000')
    started=time.monotonic()
    def serve_until(proc, seconds):
        deadline=time.monotonic()+seconds
        last_heartbeat=0.
        def heartbeat():
            nonlocal last_heartbeat
            if time.monotonic()-last_heartbeat>2.:
                write(out/'operator_heartbeat.json',dict(at=stamp(),operator_pid=os.getpid(),
                    agent_pid=proc.pid,phase=engine.phase,native_steps=engine.steps))
                last_heartbeat=time.monotonic()
        engine.on_progress=heartbeat
        while proc.poll() is None:
            heartbeat()
            if time.monotonic()>deadline:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
                engine.on_progress=None;return 'wall_limit'
            try: item=inbox.get(timeout=.2)
            except queue.Empty: continue
            payload,ready,_=item
            try:
                kind=payload['kind']
                if kind=='inspect': result=engine.inspect()
                elif kind=='observe': result={'observation':engine.observe()}
                elif kind=='step':
                    if engine.phase=='controller_validation':
                        raise RuntimeError('No robot actions allowed during controller import validation')
                    result=engine.step(payload['action'],payload.get('repeat',1))
                elif kind=='reset':
                    if engine.phase!='development': raise RuntimeError('No reset during a held-out rollout')
                    result=engine.reset()
                elif kind=='finish':
                    if engine.phase!='development': raise RuntimeError('No manual finish during a held-out rollout')
                    engine.close_episode('agent_finish'); result={'finished':True}
                elif kind=='cameras':
                    rgb,depth=engine.render()
                    result={k:{'dtype':str(v.dtype),'shape':list(v.shape),
                        'data':base64.b64encode(v.tobytes()).decode()} for k,v in {'rgb':rgb,'depth_m':depth}.items()}
                else: raise ValueError('Unknown robot operation')
                item[2]=safe(result)
            except Exception:
                item[2]={'error':traceback.format_exc(limit=5)}
            public_result=({k:v for k,v in item[2].items() if k not in ['images','rgb','depth_m']}
                if isinstance(item[2],dict) else item[2])
            record=dict(at=stamp(),request=payload,result=public_result)
            with (out/'robot_calls.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            ready.set()
        engine.on_progress=None;return 'process_exit'
    if checkpoint_from:
        write(out/'launch.json',dict(pid=os.getpid(),operator_pid=os.getpid(),started_at=stamp(),
            model=None,agent_kind='saved_checkpoint_evaluation',new_model_calls=0))
        engine.close_episode('checkpoint_validation_start_no_actions')
        decision=dict(eligible=True,status='checkpoint_validation_pending',
            kind='interrupted_agent_checkpoint',agent_finished_normally=False)
    else:
        with (out/'claude_code.jsonl').open('wb') as log,(out/'claude_stderr.log').open('wb') as err:
            proc=subprocess.Popen(args,cwd=work,env=env,stdout=log,stderr=err,start_new_session=True)
            write(out/'launch.json',dict(pid=proc.pid,operator_pid=os.getpid(),argv=args,started_at=stamp(),adaptation='CaP-X-style runtime / RAX explore-program-verify workflow',model='claude-opus-5-5',effort='max',smoke=smoke))
            stop=serve_until(proc,remaining_wall)
        engine.close_episode('agent_session_ended')
        write(out/'development_result.json',dict(ended_at=stamp(),cli_returncode=proc.returncode,
            stopping_reason=stop,wall_seconds=time.monotonic()-started,episodes=engine.closed,
            continuation=continuation,remaining_wall_at_start=remaining_wall))
        decision=agent_outcome(out/'claude_code.jsonl',proc.returncode,
            sum(e.get('steps',0) for e in engine.closed)-continuation.get('replayed_prefix_steps',0)
            ) if not smoke else dict(eligible=False,status='smoke_complete')
    if not smoke and decision['eligible']:
        # Validate the imported controller in the same isolated runtime, before spending
        # even one held-out initial state. Policy code cannot advance/reset the robot here.
        code='''import json\nimport numpy as np\nimport robot_api as robot\nfrom controller import policy\nassert callable(policy), 'controller.policy must be callable'\nobs=robot.observe()\na=np.asarray(policy(obs,0,{}),dtype=float)\nspec=json.load(open('TASK.json'))\nlo=np.asarray(spec['native_action_low']);hi=np.asarray(spec['native_action_high'])\nassert a.shape==lo.shape and np.all(np.isfinite(a)), 'invalid policy action shape/values'\nassert np.all(a>=lo-1e-7) and np.all(a<=hi+1e-7), 'policy action outside native limits'\nprint('controller import and first action validated; no physics advanced')\n'''
        (work/'preflight_controller.py').write_text(code)
        validation_env=dict(os.environ,PILOT_WORKSPACE=str(work),PILOT_OUTPUT=str(out),
            PILOT_SOCKET=str(socket_path),PYTHONPATH=str(ROOT/'src'),PILOT_READONLY_WORKSPACE='1')
        engine.phase='controller_validation'
        with (out/'controller_validation.log').open('wb') as f:
            validation=subprocess.Popen([str(PYTHON),'-c',
                'import json,sys,pilot_mcp;r=pilot_mcp.run_code("preflight_controller.py");print(json.dumps(r));sys.exit(r["returncode"])'],
                env=validation_env,stdout=f,stderr=f,start_new_session=True)
            serve_until(validation,700)
        engine.phase='development'
        if validation.returncode != 0:
            decision.update(eligible=False,status='controller_invalid',
                            reason='Controller import/first-action validation failed; held-out states not started')
    write(out/'heldout_admission.json',decision)
    evaluation_valid=True
    if not smoke and decision['eligible']:
        # Freeze every workspace file, including parameter arrays, before unseen rollouts.
        frozen=out/'frozen_workspace'; shutil.copytree(work,frozen)
        evaluation='''import sys\nimport robot_api as robot\nfrom controller import policy\nmemory={}\nobs=robot.observe()\nfor tick in range(10000):\n    action=policy(obs,tick,memory)\n    result=robot.step(action)\n    obs=result['observation']\n    if result['terminal']: break\nprint({'ticks':tick+1,'native_return':result['native_return']})\n'''
        (frozen/'evaluate.py').write_text(evaluation)
        hashes={str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in frozen.rglob('*') if p.is_file()}
        write(out/'frozen_manifest.json',hashes)
        engine.phase='heldout'; engine.index=0
        # The execution sandbox mounts this frozen snapshot as its workspace.
        eval_env=dict(os.environ,PILOT_WORKSPACE=str(frozen),PILOT_OUTPUT=str(out),PILOT_SOCKET=str(socket_path),PYTHONPATH=str(ROOT/'src'),PILOT_READONLY_WORKSPACE='1')
        for i in range(heldout):
            engine.reset()
            with (out/f'heldout_{i+1}_runner.log').open('wb') as f:
                ev=subprocess.Popen([str(PYTHON),'-c','import json,sys,pilot_mcp;r=pilot_mcp.run_code("evaluate.py");print(json.dumps(r));sys.exit(r["returncode"])'],
                    env=eval_env,stdout=f,stderr=f,start_new_session=True)
                serve_until(ev,700)
            engine.close_episode('frozen_program_ended')
            valid=ev.returncode==0 and engine.steps>0 and engine.done
            evaluation_valid=evaluation_valid and valid
            write(out/f'heldout_{i+1}_validity.json',dict(executor_returncode=ev.returncode,
                native_steps=engine.steps,native_terminated=engine.done,complete_native_rollout=valid))
        changed=[name for name,digest in hashes.items()
                 if not (frozen/name).exists() or hashlib.sha256((frozen/name).read_bytes()).hexdigest()!=digest]
        write(out/'freeze_audit.json',dict(changed_files=changed))
    final_status=('smoke_complete' if smoke else decision['status'] if not decision['eligible']
                  else ('checkpoint_completed' if checkpoint_from else 'completed')
                    if evaluation_valid else 'evaluation_incomplete')
    engine.status(completed_at=stamp(),run_status=final_status,heldout_admission=decision,
                  note='Only admitted, executed native rollouts may enter task metrics; infrastructure errors are excluded.')
    server.shutdown(); server.server_close(); socket_path.unlink(missing_ok=True)
    if engine.renderer is not None: engine.renderer.close()
    if engine.env is not None:
        engine.env.close()
        if engine.spec['benchmark']=='dm_control':engine.env.physics.free()
    faulthandler.unregister(signal.SIGUSR1);stack_file.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=TASKS);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=1);p.add_argument('--smoke',action='store_true')
    p.add_argument('--model-base-url',default='http://127.0.0.1:33622')
    p.add_argument('--cohort',default='summary_v3')
    p.add_argument('--render-backend',choices=['egl','osmesa'],default='osmesa')
    p.add_argument('--continue-from',type=Path)
    p.add_argument('--checkpoint-from',type=Path)
    p.add_argument('--resume-workspace',type=Path)
    a=p.parse_args()
    run(a.task,a.out,a.gpu,smoke=a.smoke,model_base_url=a.model_base_url,cohort=a.cohort,
        render_backend=a.render_backend,continue_from=a.continue_from,
        checkpoint_from=a.checkpoint_from,resume_workspace=a.resume_workspace)
