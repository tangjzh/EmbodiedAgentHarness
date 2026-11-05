"""Native visual-only adapters. Object state stays in the simulator process."""
import base64
import io
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageOps

TASKS = {
    'maniskill_peg': dict(benchmark='ManiSkill3', environment='PegInsertionSide-v1',
        goal='Pick up the orange-white peg and insert its orange end into the box side hole.',
        difficulty='Contact-rich side insertion with 3 mm peg/hole clearance; randomized geometry and poses.',
        horizon=100, source='https://github.com/mani-skill/ManiSkill', publication='RSS 2025'),
    'dexmimicgen_threading': dict(benchmark='DexMimicGen', environment='TwoArmThreading',
        goal='Use the two Panda arms to pick up the needle and ring-tripod and thread the needle through the ring.',
        difficulty='Bimanual relative-pose control and insertion; objects must be localized visually.',
        horizon=1000, source='https://github.com/NVlabs/dexmimicgen', publication='ICRA 2025'),
}


def plain(x):
    if hasattr(x, 'detach'): x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {k:plain(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [plain(v) for v in x]
    return x


def array(x):
    if hasattr(x, 'detach'): x = x.detach().cpu().numpy()
    return np.asarray(x)


class NativeVisualEnv:
    def __init__(self, task):
        self.task, self.spec = task, TASKS[task]
        self.env = None
        self.hz = 20

    def reset(self, seed):
        if self.env is not None: self.env.close()
        np.random.seed(seed)
        if self.task == 'maniskill_peg':
            import gymnasium as gym
            import mani_skill.envs
            from mani_skill.utils.sapien_utils import look_at
            overview = look_at([0.5,-0.6,0.6],[0.0,0.06,0.1])
            self.env = gym.make(self.spec['environment'], num_envs=1, obs_mode='rgbd',
                control_mode='pd_ee_delta_pose', reward_mode='sparse', sim_backend='cpu',
                render_backend='sapien_cuda', sensor_configs={'width':320,'height':240,
                    'base_camera':{'pose':overview}},
                render_mode='rgb_array')
            self.raw, _ = self.env.reset(seed=seed)
            self.low, self.high = self.env.action_space.low, self.env.action_space.high
            self.low, self.high = np.asarray(self.low).reshape(-1), np.asarray(self.high).reshape(-1)
            self.hz = self.env.unwrapped.control_freq
        else:
            import robosuite
            import dexmimicgen
            from robosuite import load_composite_controller_config
            self.env = robosuite.make(self.spec['environment'], robots=['Panda','Panda'],
                controller_configs=load_composite_controller_config(robot='Panda'),
                has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
                use_object_obs=False, camera_names=['agentview','robot0_eye_in_hand','robot1_eye_in_hand'],
                camera_heights=240, camera_widths=320, camera_depths=True,
                reward_shaping=False, control_freq=20, horizon=self.spec['horizon'], ignore_done=False,
                render_gpu_device_id=int(os.environ.get('ECA_RENDER_GPU','0')))
            self.raw = self.env.reset()
            self.low, self.high = self.env.action_spec
        self.steps = 0
        self.success = False
        self.done = False
        self.native_return = 0.0
        self.update_public()
        return self.public

    def update_public(self):
        self.views, cameras = {}, {}
        if self.task == 'maniskill_peg':
            # rgbd mode's extra contains only robot TCP; state mode is never enabled.
            public = {'agent':plain(self.raw['agent']), 'tcp_pose':plain(self.raw['extra']['tcp_pose'])}
            for name, data in self.raw['sensor_data'].items():
                rgb = array(data['rgb'])[0,...,:3].astype(np.uint8)
                # ManiSkill RGBD observations store millimetre depth.
                depth = array(data['depth'])[0].squeeze().astype(np.float32) / 1000.0
                self.views[name] = (rgb,depth)
                cameras[name] = plain(self.raw['sensor_param'][name])
        else:
            from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_real_depth_map
            public = {k:plain(v) for k,v in self.raw.items()
                      if k.startswith(('robot0_', 'robot1_')) and not k.endswith(('_image','_depth'))}
            for name in ['agentview','robot0_eye_in_hand','robot1_eye_in_hand']:
                # Robosuite uses an upside-down OpenGL pixel buffer. Flip RGB and depth together.
                rgb = self.raw[name+'_image'][::-1].copy()
                depth = get_real_depth_map(self.env.sim,self.raw[name+'_depth'])[::-1].squeeze().astype(np.float32)
                self.views[name] = (rgb,depth)
                cameras[name] = {'intrinsic_cv':get_camera_intrinsic_matrix(self.env.sim,name,*rgb.shape[:2]),
                                 'cam2world_cv':get_camera_extrinsic_matrix(self.env.sim,name)}
        self.public = plain(dict(robot=public, cameras=cameras, native_step=self.steps,
            control_hz=self.hz, native_success=self.success, terminal=self.done,
            native_return=self.native_return, observation_scope='RGB-D, calibration, robot proprioception only; no object GT'))

    def step(self, action):
        if self.done: raise RuntimeError('Episode ended; use reset only during development')
        a = np.asarray(action,dtype=np.float32)
        if a.shape != self.low.shape or not np.all(np.isfinite(a)):
            raise ValueError(f'Expected finite native action {self.low.shape}')
        if np.any(a<self.low-1e-6) or np.any(a>self.high+1e-6): raise ValueError('Native action outside bounds')
        if self.task == 'maniskill_peg':
            self.raw, reward, terminated, truncated, info = self.env.step(a)
            # Never expose other info fields, e.g. peg_head_pos_at_hole.
            success = bool(array(info['success']).reshape(-1)[0])
            done = bool(array(terminated).any() or array(truncated).any())
        else:
            self.raw, reward, done, info = self.env.step(a)
            success = bool(self.env._check_success())
        self.steps += 1
        self.native_return += float(array(reward).reshape(-1)[0])
        self.success = self.success or success
        # Standard success-anytime evaluation; default native horizon is unchanged.
        self.done = bool(done or self.success or self.steps >= self.spec['horizon'])
        self.update_public()
        return self.public

    def action_docs(self):
        if self.task == 'maniskill_peg':
            from mani_skill.utils import common
            c = self.env.unwrapped.agent.controller.configs
            return dict(control_mode='pd_ee_delta_pose', native_controller_config=str(c),
                layout='[dx,dy,dz, rotation-vector x,y,z, gripper]. Normalized native commands; use documented controller scales and frame.',
                quaternion_order='ManiSkill TCP quaternions are wxyz',
                camera_configuration='Default wrist RGB-D; fixed base camera repositioned for full-table overview. Native task physics, geometry, randomization and horizon unchanged.',
                depth_units='robot.cameras() depth arrays are metres; already converted from native millimetres')
        configs=[]
        for robot in self.env.robots:
            controller=robot.composite_controller.part_controllers['right']
            configs.append(dict(input_ref_frame=controller.input_ref_frame,input_type=controller.input_type,
                output_max=plain(controller.output_max),output_min=plain(controller.output_min),
                base_to_world_rotation=plain(self.env.sim.data.get_body_xmat(robot.robot_model.root_body)),
                base_origin_world=plain(self.env.sim.data.get_body_xpos(robot.robot_model.root_body))))
        return dict(control_mode='native Panda BASIC composite OSC_POSE',
            layout='robot0 [dx,dy,dz, dRx,dRy,dRz,gripper], then robot1 same; translations/axis-angle rotations are native normalized deltas in each robot base frame (see measured base-to-world rotations); +1 closes gripper, -1 opens.',
            controller_scaling=configs,
            position_delta_scale_m=0.05,rotation_delta_scale_rad=0.5,
            quaternion_order='Robosuite eef_quat uses xyzw',
            depth_units='robot.cameras() arrays are metres, with RGB and depth vertically flipped together')

    def native_audit(self):
        if self.task == 'maniskill_peg':
            return plain(self.env.unwrapped.evaluate())
        return {'native_check_success':bool(self.env._check_success())}

    def close(self):
        if self.env is not None: self.env.close()


def composite(views, step, hz):
    canvas = Image.new('RGB',(480*len(views),776),'black')
    draw = ImageDraw.Draw(canvas)
    for column,(name,(rgb,depth)) in enumerate(views.items()):
        valid = np.isfinite(depth)&(depth>0)
        d = np.zeros(depth.shape,dtype=np.uint8)
        d[valid]=(255*(1-np.clip(depth[valid]/2,0,1))).astype(np.uint8)
        for row,img in enumerate([rgb,d]):
            pic=ImageOps.contain(Image.fromarray(img).convert('RGB'),(480,360))
            x=480*column; y=row*388
            canvas.paste(pic,(x+(480-pic.width)//2,y+28+(360-pic.height)//2))
            draw.text((x+5,y+5),name+(' RGB' if row==0 else ' depth m [0,2] near=white'),fill='white')
    draw.text((6,369),f'native step {step}; {step/hz:.2f}s',fill='yellow')
    return np.asarray(canvas)


def png64(rgb):
    out=io.BytesIO();Image.fromarray(rgb).save(out,format='PNG')
    return base64.b64encode(out.getvalue()).decode()


def encoded_arrays(views):
    result={}
    for name,(rgb,depth) in views.items():
        for suffix,v in [('rgb',rgb),('depth_m',depth)]:
            result[name+'_'+suffix]=dict(dtype=str(v.dtype),shape=list(v.shape),data=base64.b64encode(v.tobytes()).decode())
    return result
