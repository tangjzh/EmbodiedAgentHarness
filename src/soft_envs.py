"""Native deformable task adapters. Public observations use a strict whitelist."""
from pathlib import Path
import sys

import numpy as np
from precision_envs import array, plain

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    'factory_nut_thread': dict(benchmark='NVIDIA Factory / Isaac Lab', environment='Isaac-Factory-NutThread-Direct-v0',
        goal='Align and screw the held M16 nut onto the bolt using the native robot controller.',
        difficulty='Thread contact, submillimetre insertion depth and coupled rotation/translation.',
        horizon=450, native_control_hz=15, metric_kind='native success-anytime',
        source='https://github.com/isaac-sim/IsaacLab',source_commit='afca7b09d60d8beb9c1cb28b43066499940b969b',
        executor_note='Native reset pre-grasps the nut; controller keeps the hand upright and clamps position relative to a noisy simulator bolt position. No object state is shown to the agent.'),
    'dlo_unknotting': dict(benchmark='DLO-Lab', environment='Train_Env_Unknotting',
        goal='Untangle the knotted rope using the two already attached Panda grippers. Maximize the official final unknotting score.',
        difficulty='Deformable rope topology, occlusion and coupled bimanual motion.',
        horizon=100, native_control_hz=50,
        source='https://github.com/UMass-Embodied-AGI/DLO-Lab',
        source_commit='4af05620a9b3c28c0fc8bd80ceb4f1edd56fd3e8',
        metric_kind='continuous; no native binary success',
        native_initialization='The environment initializes both real Panda arms grasping two predetermined rope points. Grasp discovery is not evaluated.'),
    'garment_fling': dict(benchmark='GarmentLab', environment='BimanualEnv / FlingDemo',
        goal='Use two Panda arms to pick up and fling the dress so that it lands spread out on the ground.',
        difficulty='Cloth dynamics, bimanual grasp selection, acceleration and release timing.',
        horizon=1000, native_control_hz=20,
        source='https://github.com/GarmentLab/GarmentLab',
        source_commit='6ee0620fddb8083d5eab554fa343cdc966c61d5f',
        metric_kind='native final coverage boolean and raw area ratio; evaluation resets reference cloth after the trial',
        horizon_note='The official demo defines no finite horizon. This case uses 3000 physical ticks at 60 Hz, recorded every 3 ticks (50 seconds).',
        executor_note='The native garment gripper uses PhysX attachments and filtered robot-cloth collisions. This is a simulator task, not a physical-clamp claim.'),
}


class DLOVisualEnv:
    def __init__(self, task):
        self.task, self.spec = task, TASKS[task]
        self.env = None
        self.hz = self.spec['native_control_hz']
        self.record_callback = None
        self.auto_record = False
        self.low = -np.ones(12, dtype=np.float32)
        self.high = np.ones(12, dtype=np.float32)
        self.neutral_action = np.zeros(12)

    def reset(self, seed):
        import torch
        sys.path.insert(0, str(Path(__import__('os').environ.get('ECA_DLO_ROOT',ROOT/'external/DLO-Lab'))/'experiments'))
        from omegaconf import DictConfig
        from envs.env_unknotting import Train_Env_Unknotting
        if self.env is None:
            class VisualUnknotting(Train_Env_Unknotting):
                def construct_extra_cameras(native):
                    native.rgbd_cameras = {}
                    for name, pos in [('scene_left', (-.5, 0, .65)),
                                      ('scene_right', (1.8, -.8, .7))]:
                        native.rgbd_cameras[name] = native.scene.add_camera(
                            res=(320, 240), pos=pos, lookat=(.7, -.2, .04),
                            up=(0, 0, 1), fov=45, GUI=False)
                    offset = np.diag([1., -1., -1., 1.])
                    offset[:3, 3] = [.06, 0., -.10]
                    for name, robot in [('wrist_left', native.franka1), ('wrist_right', native.franka2)]:
                        camera = native.scene.add_camera(res=(320, 240), fov=70, near=.01, GUI=False)
                        camera.attach(robot.get_link('panda_grasptarget'), offset)
                        native.rgbd_cameras[name] = camera

            self.env = VisualUnknotting(DictConfig(dict(task='unknotting',
                log_dir=str(ROOT/'runs/native_dlo_logs'), n_envs=1, n_substeps_per_step=20,
                GUI=False, camera=False, raytracer=False, seed=int(seed))))
            # Exact settings of experiments/scripts/ppo/unknotting.sh and
            # continuous_run.py; actions supplied to step_all are normalized.
            self.env.init_rl_env(n_steps=100, pos_bound=.01, angle_bound=.1, steps_interval_split=1)
            self.env.init_domain_randomization(pos_bound=(-.05, -.025, .05, .025))
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.env.reset()
        self.steps = 0
        self.done = False
        self.success = None
        self.native_return = 0.
        self.initial_score = float(self.env.reward()[0])
        self.initial_vertices = np.asarray(self.env.rope.get_all_verts())[0].copy()
        self.final_metric = None
        self.update_public()
        return self.public

    def update_public(self):
        self.views, cameras = {}, {}
        gl_to_cv = np.diag([1., -1., -1., 1.])
        for name, camera in self.env.rgbd_cameras.items():
            if name.startswith('wrist_'):
                camera.move_to_attach()
            rgb, depth, _, _ = camera.render(rgb=True, depth=True)
            rgb, depth = np.asarray(rgb), np.asarray(depth)
            if rgb.ndim == 4: rgb = rgb[0]
            if depth.ndim == 3: depth = depth[0]
            depth = depth.astype(np.float32)
            depth[(~np.isfinite(depth)) | (depth <= camera.near) | (depth >= camera.far * .999)] = 0
            self.views[name] = (rgb[..., :3].astype(np.uint8), depth)
            transform = array(camera.transform)
            if transform.ndim == 3: transform = transform[0]
            cameras[name] = dict(intrinsic_cv=[[camera.f, 0, camera.cx], [0, camera.f, camera.cy], [0, 0, 1]],
                cam2world_cv=transform @ gl_to_cv,
                pixel_centres='MSAA simultaneous RGB-D: depth u+0.375,v+0.875; nominal RGB u+0.5,v+0.5',
                depth_pixel_ray_offset=[.375,.875],rgb_pixel_ray_offset=[.5,.5],depth_units='metres')
        robot = {}
        for name, arm, ee in [('left', self.env.franka1, self.env._ef1),
                              ('right', self.env.franka2, self.env._ef2)]:
            robot[name] = dict(qpos=plain(arm.get_qpos()[0]), dof_velocity=plain(arm.get_dofs_velocity()[0]),
                eef_position_world=plain(ee.get_pos()[0]), eef_quaternion_wxyz=plain(ee.get_quat()[0]),
                gripper_attachment='native fixed rope attachment from episode initialization')
        self.public = plain(dict(robot=robot, cameras=cameras, native_step=self.steps,
            control_hz=self.hz, native_success=None, terminal=self.done,
            observation_scope='RGB-D, camera calibration and robot proprioception only; no rope/object/particle GT',
            final_native_metric=self.final_metric if self.done else None))

    def step(self, action):
        if self.done: raise RuntimeError('Episode ended')
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (12,) or not np.isfinite(a).all() or (np.abs(a) > 1+1e-6).any():
            raise ValueError('Expected 12 finite normalized commands within [-1,1]')
        import torch
        # The returned native observation contains rope GT and is deliberately discarded.
        _, reward, absorbing, _ = self.env.step_all(torch.tensor([True]), torch.as_tensor(a[None]))
        self.steps += 1
        self.native_return += float(array(reward).reshape(-1)[0])
        self.done = bool(array(absorbing).any() or self.steps >= self.spec['horizon'])
        self.update_public()
        return self.public

    def finalize_metric(self):
        final = float(self.env.reward()[0])
        self.final_metric = dict(kind='native continuous unknotting score', initial_score=self.initial_score,
            final_score=final, score_change=final-self.initial_score, native_success=None,
            rule='exp(-native unknotting_penalty); larger is better; native code defines no binary success',
            finite=bool(np.isfinite(final)))
        self.done = True
        self.public['terminal'] = True
        self.public['final_native_metric'] = self.final_metric
        return self.final_metric

    def native_audit(self):
        vertices = np.asarray(self.env.rope.get_all_verts())[0]
        initial_edges = np.linalg.norm(np.diff(self.initial_vertices, axis=0), axis=1)
        edges = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
        return plain(dict(native_score=self.env.reward(), vertices=vertices,
            max_edge_length_ratio=float(np.max(edges / np.maximum(initial_edges, 1e-12))),
            finite_vertices=bool(np.isfinite(vertices).all())))

    def action_docs(self):
        return dict(control_mode='Native DLO-Lab step_all + Pink IK + Panda joint-position tracking',
            layout='[left dx,dy,dz, right dx,dy,dz, left dRx,dRy,dRz, right dRx,dRy,dRz]',
            bounds='All 12 inputs normalized to [-1,1]. Native step_all multiplies translations by 0.01 m and rotations by 0.1 DEGREE, then clips to its original bounds. Each arm translation has native L2 cap 0.01 m.',
            translation_frame='world', rotation_convention='Native RobotControllerPink XYZ Euler increments, degrees=True; left-multiplied incremental quaternion. Quaternion observations are wxyz.',
            step_duration_seconds=.02, physics_dt=.001, physics_substeps=5,
            per_action_physics_steps=20, horizon=100, cameras='Two scene and two rigidly mounted wrist RGB-D cameras.',
            camera_convention='cam2world_cv, positive metric axial depth; simultaneous MSAA depth samples at u+0.375,v+0.875 (measured GL sample 0); nominal RGB centres u+0.5,v+0.5; no image flip needed.',
            initial_condition=self.spec['native_initialization'],
            feedback='Native continuous score is returned only after the episode ends; no dense score or rope coordinates during control.',
            limitations='Native collision/stretch failure checks are commented out upstream; physical plausibility must also be checked from video and private edge-stretch audit.')

    def close(self):
        if self.env is not None:
            self.env.scene.destroy()
            self.env = None


def make_env(task):
    if task == 'factory_nut_thread':
        from soft_factory_env import FactoryVisualEnv
        return FactoryVisualEnv(task)
    if task == 'dlo_unknotting': return DLOVisualEnv(task)
    if task == 'garment_fling':
        from soft_garment_env import GarmentVisualEnv
        return GarmentVisualEnv(task)
    raise KeyError(task)
