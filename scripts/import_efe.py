"""Import a curated morphology expansion from a fixed EFE checkout, not its solutions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '13bdc8481166ae873eb5e1eebcc51b1a81ce4e2d'
SOURCE = 'https://github.com/mll-lab-nu/embodied-first-exam.git'
SELECTION = {
    'dexhand': {'essential_cube_reorient', 'essential_pen_regrasp', 'essential_held_joystick',
                'essential_rotary_gait', 'essential_two_object', 'jar_lid_open', 'usb_plug_insert', 'ram_module_install'},
    'mujoco-playground': {'mujoco_playground/' + n for n in [
        'BarkourJoystick', 'Go1Footstand', 'Go1Getup', 'Go1Handstand', 'Go1JoystickFlatTerrain',
        'Go1JoystickRoughTerrain', 'SpotFlatTerrainJoystick', 'SpotGetup', 'SpotJoystickGaitTracking']},
    'humanoidbench': {'walk', 'balance_hard', 'hurdle', 'sit_simple', 'powerlift'},
    'embodied-olympics-fun': {'backflip', 'handstand', 'high_bar_swing', 'skateboard_ollie',
        'boxing', 'clean_and_jerk', 'dodgeball', 'trampoline', 'volleyball_serve', 'juggle_three'},
    'roboengineer': {'roboengineer/' + n for n in [
        'anycar-compact-lap-monaco-6', 'anycar-compact-lap-silverstone-8', 'anycar-drift-park',
        'anycar-tight-drift-park', 'anycar-tracking', 'g1-terrain-walk', 'go2-attitude',
        'go2-forward-jump', 'go2-weave', 'go2-yaw-jump', 'landing-random-pad',
        'neuralfly-eight', 'neuralfly-general', 'neuralfly-general-skydio', 'neuralfly-vector-wind']},
    'ambench': {'peg_in_hole_fahexa', 'press_button_fahexa', 'rotate_valve_fahexa', 'open_door_fahexa', 'ndt_fahexa'},
    'myosuite': {'baoding', 'hand_pose', 'key_turn', 'pen_twirl', 'osl_run', 'chase_tag', 'bimanual', 'table_tennis'},
    'dextoolbench': {'brush/blue_brush/sweep_forward', 'hammer/mallet_hammer/swing_down',
                     'screwdriver/long_screwdriver/spin_vertical', 'spatula/flat_spatula/serve_plate'},
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tree_hash(root):
    records = []
    for p in sorted(root.rglob('*')):
        if '.git' in p.parts or '__pycache__' in p.parts:
            continue
        relative = p.relative_to(root).as_posix()
        if p.is_symlink():
            raise ValueError('Task symlinks require explicit review: ' + str(p))
        if p.is_file():
            records.append([relative, sha(p.read_bytes()), bool(p.stat().st_mode & 0o111)])
    return sha(json.dumps(records, separators=(',', ':')).encode())


def morphology(robot):
    if robot.startswith('ambench-'):
        return 'aerial_manipulator'
    if robot in ['intel-aero-partial', 'neural-fly-partial']:
        return 'drone'
    if robot in ['unitree-go1', 'unitree-go2', 'boston-dynamics-spot', 'google-barkour-vb']:
        return 'quadruped'
    if robot.startswith('unitree-'):
        return 'humanoid'
    if robot == 'anycar':
        return 'wheeled_vehicle'
    if robot.startswith('myo'):
        return 'prosthetic_leg' if robot == 'myoleg-osl' else 'musculoskeletal'
    return 'dexterous_hand'


def identifier(suite, family):
    family = family.removeprefix('mujoco_playground/').removeprefix('roboengineer/').removeprefix('dextoolbench/')
    family = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', family).lower()
    return 'efe.' + suite.replace('-', '_') + '.' + re.sub(r'[^a-z0-9_]+', '_', family)


def variant(root, path):
    config = tomllib.loads((path / 'task.toml').read_text())
    metadata = config['metadata']
    files = ['task.toml', 'instruction.md', 'DIGEST', 'environment/Dockerfile',
             'environment/docker-compose.yaml', 'tests/test.sh', 'tests/Dockerfile']
    return dict(path=path.relative_to(root).as_posix(), tree_sha256=tree_hash(path),
        key_files={p: sha((path/p).read_bytes()) for p in files if (path/p).is_file()},
        protocol=metadata.get('protocol', 'privileged'), oracle=metadata.get('oracle', 'none'),
        upstream_validated=metadata.get('validated'), release_status=metadata.get('release_status'),
        seed=metadata.get('instance_seed', 0), seeds=metadata.get('seeds'),
        agent=config['agent'], environment=config['environment'], verifier=config.get('verifier', {}),
        dynamics=metadata.get('dynamics'), task_type=metadata.get('task_type'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'external/embodied-first-exam')
    args = p.parse_args(); source = args.root.resolve()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if head != COMMIT:
        raise ValueError('Use the reviewed EFE commit: ' + COMMIT)
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=source, text=True)
    if dirty.strip():
        raise ValueError('Import only unmodified tracked EFE source files')
    imported = []; found = Counter()
    for suite, families in SELECTION.items():
        for path in sorted((source/'tasks'/suite).glob('*-standard')):
            if not (path/'task.toml').is_file():
                continue
            spec = tomllib.loads((path/'task.toml').read_text()); meta = spec['metadata']
            family = meta['family'].removeprefix('dextoolbench/')
            if family not in families:
                continue
            if '-i01-' in path.name:
                continue
            stem = path.name.removesuffix('-standard')
            variants = {mode: variant(source, path.with_name(stem+'-'+mode))
                for mode in ['standard', 'privileged', 'realtime']
                if (path.with_name(stem+'-'+mode)/'task.toml').is_file()}
            robot = meta['robot']; kind = morphology(robot)
            label = family.removeprefix('mujoco_playground/').removeprefix('roboengineer/')
            label = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', label).replace('_', ' ').replace('-', ' ')
            imported.append(dict(id=identifier(suite, family), name=label,
                benchmark='Embodied First Exam / '+suite, suite=suite, robot=robot,
                morphology=kind, hands=meta.get('hands', []), family=family, task_type=meta['task_type'],
                backend='efe', group='efe', enabled=True, default_mode='standard', modes=variants,
                goal=spec['task']['description'], effort='high', wall_seconds=int(spec['agent']['timeout_sec']),
                development_episodes=None, heldout_episodes=None, max_turns=None,
                observation=('Noisy deployment observations and a public nominal training simulator; '
                    'data-only policy submission through the native fixed ABI.' if meta['task_type']=='policy' else
                    'Task-declared sensors in Standard; full deployment simulator access requires explicit Privileged mode.'),
                historical_result={'status': 'not_run_in_this_repository'},
                integration_status='catalog_imported',
                setup_notes=['RoboEngineer evaluator images must be built locally.' if suite=='roboengineer' else
                    'Install/pull the pinned native simulator and files images.',
                    'Neural Fly requires its local force-fit asset.' if 'neuralfly' in path.name else
                    'No prior task code or reference solution is exposed to a developing agent.']))
            found[suite] += 1
    if len(imported) != sum(len(v) for v in SELECTION.values()):
        raise ValueError('Selection mismatch: ' + str(found))
    if len({x['id'] for x in imported}) != len(imported):
        raise ValueError('Task identifiers collide')
    runtime = ['scripts/stamp_mtimes.py', 'scripts/task_seed.py', 'scripts/efe.py',
               'scripts/efe_harbor_hook.py', 'scripts/efe-harbor.pth']
    result = dict(schema_version=1, source=SOURCE, commit=COMMIT, harbor_version='0.23.0',
        agent_cli_version='2.1.304',
        default_mode='standard', no_cross_task_transfer=True,
        runtime_files={p: sha((source/p).read_bytes()) for p in runtime}, tasks=imported)
    target=ROOT/'tasks/efe_catalog.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(tasks=len(imported),mode_variants=sum(len(t['modes']) for t in imported),
        morphologies=dict(Counter(x['morphology'] for x in imported)), robots=len({x['robot'] for x in imported}),
        suites=dict(found)), ensure_ascii=False))


if __name__ == '__main__':
    main()
