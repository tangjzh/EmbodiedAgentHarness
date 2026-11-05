"""Enforce nonprivileged S2/S3/M3/M4 observations in the released harness.

The released simulator unconditionally includes cube truth and the code
executor injects the simulator itself. These are shared boundary corrections,
not changes to perception, IK, motion, task goals or task grading.
"""
import ast
import json


PUBLIC_PROPRIO = {'robot_cartesian_pos', 'robot_joint_pos', 'robot_cartesian_pose_wxyz_xyz',
    'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel',
    'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel',
    'robot0_proprio-state'}
IMPORTS = {'numpy', 'math', 'scipy', 'scipy.optimize', 'scipy.spatial', 'scipy.spatial.transform',
           'viser', 'viser.transforms', 'open3d'}


def public_observation(obs, cameras):
    return {k:v for k,v in obs.items() if k in PUBLIC_PROPRIO or k in cameras}


def validate(code):
    tree=ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node,ast.Name) and (node.id in {'env','APIS','exec','eval','open','compile','globals','locals','vars','__import__'} or node.id.startswith('__')):
            raise ValueError('Simulator, private executor and external I/O are outside the nonprivileged interface')
        if isinstance(node,ast.Attribute) and (node.attr.startswith('_') or node.attr in {'load','save','fromfile','tofile','memmap','ctypes','check_success','compute_reward'}):
            raise ValueError('Private state and external I/O attributes are unavailable')
        if isinstance(node,ast.Import) and any(n.name not in IMPORTS for n in node.names):
            raise ValueError('Use only numerical and documented geometry libraries')
        if isinstance(node,ast.ImportFrom) and (node.module not in IMPORTS or node.level):
            raise ValueError('Use only numerical and documented geometry libraries')
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in {'getattr','hasattr'}:
            if len(node.args)>1 and isinstance(node.args[1],ast.Constant) and str(node.args[1].value).startswith('_'):
                raise ValueError('Private attributes are unavailable')
    return tree


def install(env, out):
    low=env.low_level_env
    observe=low.get_observation
    low.get_observation=lambda: public_observation(observe(),low.render_camera_names)
    execute=env._exec_user_code
    def guarded(code):
        try:
            validate(code)
        except (ValueError,SyntaxError) as exc:
            message='InterfaceViolation: '+str(exc)
            with (out/'interface_violations.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(error=message,code=code))+'\n')
            return dict(ok=False,stdout='',stderr=message)
        return execute(code)
    env._exec_user_code=guarded
