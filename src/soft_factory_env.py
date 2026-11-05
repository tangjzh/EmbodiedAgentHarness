"""Native NVIDIA Factory assembly with visual observations only."""
import numpy as np
from precision_envs import array, plain
from soft_envs import TASKS


class FactoryVisualEnv:
    def __init__(self, task):
        self.task,self.spec=task,TASKS[task]
        self.check_rotation=task=='factory_nut_thread'
        self.env=None;self.app=None;self.record_callback=None;self.auto_record=False
        self.hz=15;self.per_action_physics_steps=8
        self.low=-np.ones(6,dtype=np.float32);self.high=np.ones(6,dtype=np.float32)
        # No requested translation, minimum raw yaw input. Upstream updates the
        # smoothed yaw in place during decimation; this is NOT a zero-motion hold.
        self.neutral_action=np.array([0.,0.,0.,0.,0.,-1. if self.check_rotation else 0.])

    def reset(self,seed):
        if self.app is None:
            from isaaclab.app import AppLauncher
            launcher=AppLauncher(dict(headless=True,enable_cameras=True,device='cuda:0',multi_gpu=False))
            self.app=launcher.app
            import torch
            from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
            from isaaclab_tasks.direct.factory import factory_env_cfg
            from isaaclab.sensors import Camera,CameraCfg
            import isaaclab.sim as sim_utils

            class VisualFactory(FactoryEnv):
                _inside_control_step=False
                suppressed_terminal_resets=0

                def _setup_scene(native):
                    super()._setup_scene()
                    native.visual_cameras={}
                    for name,path,size,offset in [
                        ('scene','/World/envs/env_.*/visual_scene',(320,240),CameraCfg.OffsetCfg()),
                        ('closeup','/World/envs/env_.*/visual_closeup',(640,480),CameraCfg.OffsetCfg()),
                        ('wrist','/World/envs/env_.*/Robot/panda_fingertip_centered/visual_wrist',(640,480),
                            CameraCfg.OffsetCfg(pos=(.10,0.,-.08),
                                rot=(.9238795325,0.,-.3826834324,0.),convention='ros'))]:
                        sensor=Camera(CameraCfg(prim_path=path,update_period=0.,height=size[1],width=size[0],
                            data_types=['rgb','distance_to_image_plane'],update_latest_camera_pose=True,
                            spawn=sim_utils.PinholeCameraCfg(focal_length=18.,horizontal_aperture=24.,clipping_range=(.005,3.)),
                            offset=offset))
                        native.visual_cameras[name]=sensor
                        native.scene.sensors['visual_'+name]=sensor

                def _reset_idx(native,env_ids):
                    if native._inside_control_step:
                        # Preserve terminal physics, preventing DirectRLEnv's
                        # automatic reset from inserting an uncounted episode.
                        native.suppressed_terminal_resets+=1
                        return
                    return super()._reset_idx(env_ids)

            cfg=getattr(factory_env_cfg,self.spec.get('factory_configuration','FactoryTaskNutThreadCfg'))()
            cfg.scene.num_envs=1;cfg.sim.device='cuda:0';cfg.seed=int(seed)
            cfg.sim.render_interval=cfg.decimation
            self.env=VisualFactory(cfg,render_mode='rgb_array')
            assert self.env.max_episode_length==self.spec['horizon'],(self.env.max_episode_length,cfg.sim.dt,cfg.decimation)
        import torch
        self.env._inside_control_step=False
        self.env.reset(seed=int(seed))
        self.env.visual_cameras['scene'].set_world_poses_from_view(
            eyes=torch.tensor([[.95,-.45,.45]],device=self.env.device),
            targets=torch.tensor([[.6,0.,.04]],device=self.env.device))
        self.env.visual_cameras['closeup'].set_world_poses_from_view(
            eyes=torch.tensor([[.6,.24,.24]],device=self.env.device),
            targets=torch.tensor([[.6,0.,.035]],device=self.env.device))
        for _ in range(8):self.env.sim.render()
        self.steps=0;self.done=False;self.success=False;self.native_return=0.;self.final_metric=None
        self.update_public()
        return self.public

    def update_public(self):
        from scipy.spatial.transform import Rotation
        self.views={};calibration={}
        for name,camera in self.env.visual_cameras.items():
            camera.update(0.,force_recompute=True)
            data=camera.data
            rgb=array(data.output['rgb'])[0,...,:3].astype(np.uint8)
            depth=array(data.output['distance_to_image_plane'])[0].squeeze().astype(np.float32)
            depth[(~np.isfinite(depth))|(depth<=.005)|(depth>=2.999)]=0
            self.views[name]=(rgb,depth)
            q=array(data.quat_w_ros)[0]
            transform=np.eye(4)
            transform[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            transform[:3,3]=array(data.pos_w)[0]
            calibration[name]=dict(intrinsic_cv=plain(data.intrinsic_matrices[0]),cam2world_cv=transform,
                pixel_centres='u+0.5,v+0.5 (verified rendered-depth ray centres)',
                depth_pixel_ray_offset=[.5,.5],rgb_pixel_ray_offset=[.5,.5],depth_units='metres')
        self.public=plain(dict(robot=dict(joint_positions=self.env._robot.data.joint_pos[0],
            joint_velocities=self.env._robot.data.joint_vel[0],applied_joint_torques=self.env._robot.data.applied_torque[0],
            eef_position_world=self.env.fingertip_midpoint_pos[0],eef_quaternion_wxyz=self.env.fingertip_midpoint_quat[0]),
            cameras=calibration,native_step=self.steps,control_hz=self.hz,terminal=self.done,
            native_success=self.success,final_native_metric=self.final_metric if self.done else None,
            observation_scope='RGB-D, calibrated cameras and robot proprioception only; native policy/critic object states are discarded'))

    def step(self,action):
        if self.done:raise RuntimeError('Episode ended')
        a=np.asarray(action,dtype=np.float32)
        if a.shape!=(6,) or not np.isfinite(a).all() or (np.abs(a)>1+1e-6).any():
            raise ValueError('Expected six finite normalized native actions in [-1,1]')
        import torch
        self.env._inside_control_step=True
        try:
            # Both returned observations contain object GT and are discarded.
            _,reward,terminated,truncated,_=self.env.step(torch.as_tensor(a[None],device=self.env.device))
        finally:self.env._inside_control_step=False
        self.steps+=1;self.native_return+=float(array(reward).reshape(-1)[0])
        current=bool(array(self.env._get_curr_successes(self.env.cfg_task.success_threshold,check_rot=self.check_rotation))[0])
        self.success=bool(self.success or current or array(self.env.ep_succeeded)[0])
        self.done=bool(array(terminated).any() or array(truncated).any() or self.steps>=self.spec['horizon'])
        self.update_public()
        return self.public

    def finalize_metric(self):
        final_success=bool(array(self.env._get_curr_successes(
            self.env.cfg_task.success_threshold,check_rot=self.check_rotation))[0])
        self.final_metric=dict(kind='native Factory '+self.env.cfg_task.name+' success-anytime',native_success=self.success,
            native_final_success=final_success,success_anytime=self.success,
            control_steps=self.steps,native_horizon=self.spec['horizon'],
            native_success_rule=self.spec.get('success_rule','Held base XY within 2.5 mm, height within 0.375 thread pitch of native target, and EEF yaw below native success yaw.'),
            suppressed_terminal_autoresets=self.env.suppressed_terminal_resets)
        self.done=True;self.public['terminal']=True;self.public['final_native_metric']=self.final_metric
        return self.final_metric

    def native_audit(self):
        return plain(dict(native_success_now=self.env._get_curr_successes(self.env.cfg_task.success_threshold,check_rot=self.check_rotation),
            held_position=self.env.held_pos,held_quaternion=self.env.held_quat,
            fixed_position=self.env.fixed_pos,fixed_quaternion=self.env.fixed_quat,
            ep_succeeded=self.env.ep_succeeded,suppressed_terminal_autoresets=self.env.suppressed_terminal_resets))

    def action_docs(self):
        docs=dict(control_mode='Native NVIDIA Factory Cartesian impedance/torque executor',
            layout='[dx,dy,dz,dRx,dRy,yaw_command], normalized [-1,1]. EMA smoothing 0.2; incremental world XYZ scale 0.02 m; rotation scale 0.097 rad. Native yaw remaps inputs toward clockwise rotation. The upstream implementation modifies smoothed yaw in-place in each physics substep: -1 is the minimum raw yaw request, NOT a reliable zero-yaw hold. Use measured EEF orientation feedback.',
            control_limits='Native executor forces upright roll=pi,pitch=0, closes the gripper continuously, and clamps motion within 5 cm of the noisy fixed-asset position. These built-in restrictions use simulator object position inside the executor; that position is NOT exposed to the coding agent.',
            initial_condition='Native randomized reset places the nut in a closed gripper near the bolt. Alignment and threading are evaluated; finding/grasping a loose nut is not.',
            step_duration_seconds=1/15,physics_dt=1/120,per_action_physics_steps=8,
            horizon=450,native_horizon_seconds=30.,quaternion_order='wxyz',
            camera_convention='cam2world_cv using ROS optical axes x right,y down,z forward; rendered depth rays use u+0.5,v+0.5; axial metric depth.',
            feedback='Only robot proprioception, RGB-D and native success. Object GT, native policy/critic observations and dense rewards stay operator-private.',
            terminal_handling='Native automatic reset is suppressed only at terminal control steps; terminal physics state is retained and the 450-step horizon is unchanged.')
        if not self.check_rotation:
            docs.update(layout='[dx,dy,dz,dRx,dRy,dRz], normalized [-1,1]. EMA smoothing 0.2; incremental world XYZ scale 0.02 m; axis-angle rotation scale 0.097 rad. Native executor restricts roll/pitch; measure actual orientation response. No forced clockwise yaw remapping for this task.',
                initial_condition='Native randomized reset pre-grasps the held assembly part. Finding and grasping a loose part is not evaluated.',
                horizon=self.spec['horizon'],native_horizon_seconds=self.spec['horizon']/15,
                terminal_handling='Native automatic reset is suppressed only at terminal control steps; terminal physics and the native task horizon are retained.')
        if self.env is not None:
            fields=['diameter','height','mass','friction','base_height','thread_pitch']
            docs['nominal_asset_properties']={name:{k:getattr(asset,k) for k in fields if hasattr(asset,k)}
                for name,asset in [('held',self.env.cfg_task.held_asset_cfg),('fixed',self.env.cfg_task.fixed_asset_cfg)]}
        return docs

    def close(self):
        if self.env is not None:self.env.close()
        if self.app is not None:self.app.close()
