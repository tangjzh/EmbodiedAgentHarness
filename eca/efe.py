"""Pinned EFE tasks, executed by unmodified Harbor installed agents."""
import hashlib
import fnmatch
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlsplit
from .catalog import ROOT


def manifest():
    return json.loads((ROOT / 'tasks/efe_catalog.json').read_text())


def source_root():
    return Path(os.environ.get('ECA_EFE_ROOT', ROOT / 'external/embodied-first-exam')).resolve()


def harbor_binary():
    return str(Path(os.environ.get('ECA_EFE_HARBOR', ROOT / '.venvs/efe/bin/harbor')).resolve())


def tree_hash(root):
    records = []
    for path in sorted(root.rglob('*')):
        if '.git' in path.parts or '__pycache__' in path.parts:
            continue
        if path.is_symlink():
            raise ValueError('Unreviewed task symlink: ' + str(path))
        if path.is_file():
            records.append([path.relative_to(root).as_posix(),
                hashlib.sha256(path.read_bytes()).hexdigest(), bool(path.stat().st_mode & 0o111)])
    return hashlib.sha256(json.dumps(records, separators=(',', ':')).encode()).hexdigest()


def selected(item, args):
    mode = getattr(args, 'efe_mode', 'standard')
    if mode not in item['modes']:
        raise ValueError(f"{item['id']} has no upstream {mode} mode; available: {', '.join(item['modes'])}")
    if getattr(args, 'wall_seconds', None) is not None:
        raise ValueError('EFE preserves task-native timeouts. Omit --wall-seconds.')
    spec = item['modes'][mode]
    if args.command == 'replay' and spec['oracle'] == 'none':
        raise ValueError('Upstream declares no oracle for this task/mode; use develop or qualify.')
    return mode, spec


def native_command(item, args, staged, jobs):
    selected(item, args)
    agent = {'develop': 'claude-code', 'replay': 'oracle', 'qualify': 'nop'}[args.command]
    cmd = [harbor_binary(), 'run', '-p', str(staged), '-a', agent,
           '-n', '1', '-k', '1', '-r', '0', '-y', '-o', str(jobs), '--job-name', 'trial',
           '-e', getattr(args, 'engine', 'docker')]
    if args.command == 'develop':
        cmd += ['-m', args.model, '--ak', 'reasoning_effort=' + item['effort'],
                '--ak', 'version=' + manifest()['agent_cli_version'],
                '--ak', 'disallowed_tools=WebSearch,WebFetch']
        if getattr(args, 'allow_model_host', False):
            host = urlsplit(args.base_url).hostname
            if not host:
                raise ValueError('Invalid model base URL')
            cmd += ['--allow-agent-host', host]
    return cmd


def plan(item, args):
    mode, spec = selected(item, args)
    out = Path(args.out).resolve()
    return dict(task=item['id'], mode=args.command, backend='efe', efe_mode=mode,
        agent={'develop': 'claude-code', 'replay': 'oracle', 'qualify': 'nop'}[args.command],
        model=args.model if args.command == 'develop' else None,
        agent_cli_version=manifest()['agent_cli_version'] if args.command == 'develop' else None,
        effort=item['effort'], initialization='empty_task_workspace' if args.command == 'develop' else args.command,
        no_cross_task_transfer=True, reference_visible_to_agent=False,
        wall_seconds=spec['agent']['timeout_sec'], agent_budget=spec['agent'],
        environment_budget=spec['environment'], protocol=spec['protocol'], task_type=spec['task_type'],
        oracle_coverage=spec['oracle'], release_status=spec['release_status'],
        upstream_commit=manifest()['commit'], task_tree_sha256=spec['tree_sha256'],
        observation=('Full simulator access as declared by the upstream Privileged task.' if mode == 'privileged'
                     else item['observation']),
        output=str(out), environment_group='efe', historical_result=item['historical_result'],
        official_leaderboard_submission=False,
        model_transport_extra_host=(urlsplit(args.base_url).hostname
            if args.command == 'develop' and getattr(args, 'allow_model_host', False) else None),
        command=native_command(item, args, out/'staged'/Path(spec['path']).name, out/'jobs'))


def verify_source(item, args):
    _, spec = selected(item, args)
    root = source_root()
    if not (root / '.git').exists():
        raise ValueError('Missing EFE checkout. Run: eca setup --group efe --execute')
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    expected = manifest()
    if head != expected['commit']:
        raise ValueError('EFE commit mismatch: expected ' + expected['commit'] + ', got ' + head)
    for relative, digest in expected['runtime_files'].items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('EFE runtime changed: ' + relative)
    task_dir = root / spec['path']
    if not task_dir.is_dir() or tree_hash(task_dir) != spec['tree_sha256']:
        raise ValueError('EFE task files changed: ' + spec['path'])
    return root, task_dir


def runtime_check(binary):
    python = Path(binary).parent / 'python'
    if not Path(binary).is_file() or not python.is_file():
        raise ValueError('Missing isolated EFE runtime. Run: eca setup --group efe --execute')
    probe = subprocess.run([str(python), '-c',
        'import json,sys,hashlib,efe; from importlib.metadata import version; from pathlib import Path; '
        'print(json.dumps([list(sys.version_info[:2]), version("harbor"), '
        'version("embodied-first-exam"), hashlib.sha256(Path(efe.__file__).read_bytes()).hexdigest()]))'],
        capture_output=True, text=True, check=True, timeout=30)
    py, harbor, efe_version, wrapper_sha = json.loads(probe.stdout)
    if not ([3, 12] <= py < [3, 15]) or harbor != manifest()['harbor_version']:
        raise ValueError('EFE requires Python 3.12–3.14 and Harbor ' + manifest()['harbor_version'])
    if wrapper_sha != manifest()['runtime_files']['scripts/efe.py']:
        raise ValueError('Installed EFE wrapper differs from the pinned checkout. Re-run EFE setup.')
    return {'python': py, 'harbor': harbor, 'embodied_first_exam': efe_version}


def check_endpoint(item, args):
    if args.command != 'develop':
        return
    _, spec = selected(item, args)
    host = urlsplit(args.base_url).hostname
    if not host:
        raise ValueError('Invalid model base URL')
    budget = spec['agent']
    if budget.get('network_mode') == 'allowlist':
        if not any(fnmatch.fnmatchcase(host.lower(), pattern.lower())
                   for pattern in budget.get('allowed_hosts', [])) and not getattr(args, 'allow_model_host', False):
            raise ValueError('Model endpoint is outside this EFE task network allowlist: ' + host +
                             '. Select an allowed endpoint; task network permissions are preserved.')


def run(item, args):
    description = plan(item, args)
    check_endpoint(item, args)
    root, task_dir = verify_source(item, args)
    description['runtime'] = runtime_check(harbor_binary())
    engine = getattr(args, 'engine', 'docker')
    if not shutil.which(engine):
        raise ValueError('Missing container engine: ' + engine)
    # Read-only preflight before consuming a new output path or launching a trial.
    info = subprocess.run([engine, 'info'], capture_output=True, text=True, timeout=30)
    if info.returncode:
        raise ValueError(engine + ' daemon is unavailable to this user; run doctor on the rendering host.')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'run.json').write_text(json.dumps(description, ensure_ascii=False, indent=2)+'\n')
    env = dict(os.environ, EFE_ROOT=str(root))
    # Docker Compose's Go client assumes /tmp is writable; some shared hosts restrict it.
    # A host-local directory changes no task image, budget, or observation permission.
    (out/'host-tmp').mkdir(mode=0o700)
    env['TMPDIR'] = str(out/'host-tmp')
    if args.command == 'develop':
        env['ANTHROPIC_BASE_URL'] = args.base_url
    # Upstream's content-derived mtimes prevent cross-task BuildKit cache collisions.
    subprocess.run([str(Path(harbor_binary()).parent/'python'),
        str(root/'scripts/stamp_mtimes.py'), 'stage', str(task_dir), str(out/'staged')],
        env=env, text=True, capture_output=True, check=True)
    staged = out/'staged'/task_dir.name
    if tree_hash(staged) != description['task_tree_sha256']:
        raise ValueError('Staged task content differs from the pinned source')
    command = native_command(item, args, staged, out/'jobs')
    with (out/'runner.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line); log.flush()
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate(); proc.wait(timeout=30)
            code = 130
    finish(out, code)


def finish(out, code):
    from .summary import harbor_trials
    trials = harbor_trials(out)
    graded = sum(t['status'] in ['graded', 'exception_with_verdict'] for t in trials)
    (out/'runner_status.json').write_text(json.dumps({'returncode': code,
        'trial_count': len(trials), 'graded_trials': graded})+'\n')
    if code:
        raise RuntimeError(f'Native Harbor exited {code}; inspect {out}/runner.log and native trial results.')
    if not trials or graded != len(trials):
        raise RuntimeError(f'Native Harbor produced an infrastructure error or no graded verdict; inspect {out}/jobs. This is not a task failure.')


def inspect(item):
    from types import SimpleNamespace
    checks = []
    def check(name, passed, detail):
        checks.append(dict(name=name, passed=bool(passed), detail=detail))
    args = SimpleNamespace(command='qualify', efe_mode='standard', wall_seconds=None)
    try:
        root, task_dir = verify_source(item, args)
        check('pinned source and task hashes', True, str(task_dir))
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        check('pinned source and task hashes', False, str(e))
    try:
        check('isolated native runtime', True, runtime_check(harbor_binary()))
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        check('isolated native runtime', False, str(e))
    binary = shutil.which('docker')
    try:
        probe = subprocess.run(['docker', 'info'], capture_output=True, timeout=30) if binary else None
        check('Docker daemon', probe and probe.returncode == 0, 'Requires Docker access on the execution host')
        compose = subprocess.run(['docker', 'compose', 'version'], capture_output=True, timeout=30) if binary else None
        check('Docker Compose', compose and compose.returncode == 0, 'Native task sidecars use Compose')
    except (OSError, subprocess.SubprocessError) as e:
        check('container runtime', False, str(e))
    return dict(task=item['id'], passed=all(c['passed'] for c in checks), checks=checks,
                paid_model_calls=0, note='Does not pull images, check task assets, or qualify the simulator.')


def setup(args):
    import shlex
    root = source_root()
    venv = (args.venv or ROOT/'.venvs/efe').resolve()
    commands = []
    if not (root/'.git').exists():
        if root.exists() and any(root.iterdir()):
            raise ValueError('Preserve existing non-Git EFE directory: ' + str(root))
        commands.append(['git', 'clone', '--filter=blob:none', '--no-checkout', manifest()['source'], str(root)])
    else:
        dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True)
        if dirty.strip():
            raise ValueError('Preserve modified EFE checkout; select a clean ECA_EFE_ROOT.')
    suites = sorted({t['suite'] for t in manifest()['tasks']})
    commands += [['git', '-C', str(root), 'sparse-checkout', 'init', '--cone'],
        ['git', '-C', str(root), 'sparse-checkout', 'set', 'scripts', 'docs', 'ci_checks', 'images'] +
            ['tasks/'+s for s in suites],
        ['git', '-C', str(root), 'checkout', '--detach', manifest()['commit']]]
    if not (venv/'bin/python').is_file():
        commands.append(['uv', 'venv', '--python', '3.12', str(venv)])
    commands.append(['uv', 'pip', 'install', '--python', str(venv/'bin/python'), '--editable', str(root)])
    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)
    print('EFE uses its own Python 3.12 / Harbor 0.23.0 environment. '
          'Docker images and suite-specific assets are separate; see docs/EFE.md.')
