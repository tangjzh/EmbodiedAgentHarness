"""Expose the UV interpreter runtime read-only inside the upstream bundle sandbox.

The upstream venv uses a symlink outside its own prefix. No agent workspace,
host home directory, credentials, network, or additional robot APIs are exposed.
"""
import os
from pathlib import Path
import subprocess
import sys

from services.controller.sandbox import BundleSandbox


class UvBundleSandbox(BundleSandbox):
    def command(self, *args, **kwargs):
        command = super().command(*args, **kwargs)
        boundary = command.index('--')
        prefix = str(Path(sys.base_prefix).resolve())
        mounts = ['--ro-bind', prefix, prefix]
        executable = Path(sys.prefix) / 'bin/python'
        if executable.is_symlink():
            target = Path(os.readlink(executable))
            if target.is_absolute() and target.resolve().is_relative_to(Path(prefix)):
                alias = str(target.parent.parent)
                if alias != prefix:
                    mounts += ['--ro-bind', prefix, alias]
        command[boundary:boundary] = mounts
        return command


def spawn_uv_worker(parent_root, bundle_id, index, connection):
    from services.controller.formal_worker import PROJECT, worker_root
    root = worker_root(parent_root, index)
    root.parent.mkdir(mode=0o700, exist_ok=True)
    environment = {key: value for key, value in os.environ.items() if key != 'OPENROUTER_API_KEY'}
    environment['PYTHONPATH'] = str(Path(__file__).resolve().parent) + ':' + str(PROJECT)
    shim = ('from functools import partial; '
            'import services.controller.formal_worker as worker; '
            'from rax_runtime_compat import UvBundleSandbox; '
            'worker.run_worker = partial(worker.run_worker, runner_factory=UvBundleSandbox); '
            'worker.main()')
    with (root.parent / f'{root.name}.log').open('xb') as log:
        return subprocess.Popen([sys.executable, '-c', shim,
            '--parent-root', str(parent_root), '--bundle', bundle_id, '--index', str(index),
            '--channel-fd', str(connection.fileno()), '--parent-pid', str(os.getpid())],
            pass_fds=(connection.fileno(),), stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, cwd=PROJECT, env=environment, start_new_session=True)
