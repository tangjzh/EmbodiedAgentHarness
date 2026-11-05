"""Fresh Claude Code over fixed upstream CaP-X APIs."""
import json
import os
from pathlib import Path
import sys
import numpy as np
from .catalog import ROOT
class CapxVisualEnv:
    def __init__(self,item):
        self.item=item;self.task=item['id'];self.spec=dict(item,horizon=4000)
        self.native=None;self.record_callback=None;self.auto_record=True;self.hz=20
        self.low=np.zeros(0);self.high=np.zeros(0);self.neutral_action=np.zeros(0)
    def reset(self,seed):
        if self.native is None:self.build()
        self.steps=0;self.native.reset(seed=int(seed));self.steps=0
        self.done=False;self.success=False;self.native_return=0.;self.final_metric=None;self.last_execution={}
        self.update_public();return self.public
    def build(self):
        root=Path(os.environ.get('ECA_CAPX_ROOT',ROOT/'external/cap-x'))
        sys.path.insert(0,str(root))
        dependency='LIBERO-PRO' if self.item['group']=='capx_libero' else 'robosuite'
        sys.path.insert(0,str(root/'capx/third_party'/dependency))
        import capx.integrations
        import capx.envs.simulators
        import capx.integrations.vision.sam3 as sam
        import capx.integrations.vision.graspnet as grasp
        from capx.integrations.motion.pyroki import init_pyroki
        sam.SERVICE_URL=os.environ.get('ECA_SAM3_URL','http://127.0.0.1:8114')
        grasp.SERVICE_URL=os.environ.get('ECA_GRASPNET_URL','http://127.0.0.1:8115')
        for name,module in list(sys.modules.items()):
            if name.startswith('capx.integrations') and hasattr(module,'init_pyroki'):
                module.init_pyroki=lambda:init_pyroki(server_url=os.environ.get('ECA_IK_URL','http://127.0.0.1:8116'))
        from capx.envs.tasks.base import CodeExecEnvConfig
        if self.item['group']=='capx_libero':
            from capx.envs.simulators.libero import FrankaLiberoEnv
            from capx.envs.tasks.franka.franka_libero_env import FrankaLiberoCodeEnv
            low=FrankaLiberoEnv(suite_name=self.item['suite'],task_id=self.item['task_id'],
                privileged=False,max_steps=4000,enable_render=False)
            self.native=FrankaLiberoCodeEnv(CodeExecEnvConfig(low_level=low,apis=['FrankaLiberoApi'],
                prompt='Goal: '+self.item['goal'],privileged=False,enable_render=False))
        else:
            from capx.envs.tasks.franka.franka_lift import FrankaLiftCodeEnv
            from capx.envs.tasks.franka.franka_pick_place import FrankaPickPlaceCodeEnv
            lift=self.item['id']=='cube_lift'
            cls=FrankaLiftCodeEnv if lift else FrankaPickPlaceCodeEnv
            self.native=cls(CodeExecEnvConfig(low_level='franka_robosuite_cube_lift_low_level' if lift else 'franka_robosuite_cube_stack_low_level',
                apis=['FrankaControlApi'],privileged=False,enable_render=False))
        low=self.native.low_level_env
        from capx_boundary import install
        initial=low.get_observation()
        if not hasattr(low,'render_camera_names'):
            low.render_camera_names=[k for k,v in initial.items() if isinstance(v,dict) and 'images' in v]
        install(self.native,self.output)
        simulator=getattr(low,'robosuite_env',None)
        if simulator is None:simulator=low.handle.env
        self.simulator=simulator;step=simulator.step
        def recorded(*args,**kwargs):
            value=step(*args,**kwargs);self.steps=getattr(self,'steps',0)+1
            if self.record_callback:self.update_public();self.record_callback()
            return value
        simulator.step=recorded
    def update_public(self):
        from precision_envs import plain
        obs=self.native.low_level_env.get_observation();self.views={};public={}
        for key,value in obs.items():
            if isinstance(value,dict) and 'images' in value:
                images=value['images'];rgb=np.asarray(images['rgb'])[...,:3].astype(np.uint8)
                depth=np.asarray(images['depth']).squeeze().astype(np.float32)
                self.views[key]=(rgb,depth)
                public[key]={k:plain(v) for k,v in value.items() if k!='images'}
            else:public[key]=plain(value)
        self.public=dict(**public,native_step=self.steps,control_hz=20,terminal=getattr(self,'done',False),
            native_success=getattr(self,'success',False),capx_api_prompt=self.native._full_prompt,
            last_execution=getattr(self,'last_execution',{}))
    def execute_command(self,operation,**arguments):
        if operation!='execute_code':raise ValueError('Use execute_code with a code string')
        _,reward,terminated,truncated,info=self.native.step(arguments['code'])
        self.success=bool(info.get('task_completed'));self.done=bool(terminated or truncated);self.native_return=float(reward)
        self.last_execution={k:info[k] for k in ['stdout','stderr','sandbox_rc'] if k in info};self.update_public()
    def action_docs(self):
        return dict(interface="robot.command('execute_code', code=python_source_string)",
            instruction='Read capx_api_prompt. controller.py defines main() and sends CaP-X code through robot.command. Fixed APIs; no env/APIS/private state or external I/O. Numeric robot.step is unavailable.')
    def finalize_metric(self):
        self.success=bool(self.native.low_level_env.task_completed())
        self.final_metric=dict(native_final_success=self.success);return self.final_metric
    def close(self):
        if self.native is not None:self.simulator.close()
def run(item,args,out,base):
    import soft_envs,soft_pilot
    soft_envs.TASKS[item['id']]=dict(item,horizon=4000,native_control_hz=20,
        metric_kind='native task completion',difficulty='Fixed upstream CaP-X task interface')
    original=soft_pilot.make_env
    def make(task):
        if task==item['id']:
            env=CapxVisualEnv(item);env.output=out;return env
        return original(task)
    soft_pilot.make_env=make
    if args.command=='replay':
        raise ValueError('Historical interfaces differ. Archived code is supplied for inspection, but exact legacy replay is not yet validated. Use develop for a fresh Claude Code run.')
    if args.command=='qualify':
        out.mkdir(parents=True,exist_ok=False);env=make(item['id'])
        try:
            env.reset(307000)
            checks={k:dict(rgb_std=float(rgb.std()),finite_depth_fraction=float(np.isfinite(d).mean())) for k,(rgb,d) in env.views.items()}
            if not checks or any(v['rgb_std']<=5 for v in checks.values()):raise RuntimeError('Camera qualification failed')
            (out/'qualification.json').write_text(json.dumps(dict(passed=True,paid_model_calls=0,cameras=checks,
                scope='Camera initialization only; perception/IK services need separate checks'),indent=2))
        finally:env.close()
        return
    from .launch import arguments
    soft_pilot.run(arguments(item,args,out,base))
