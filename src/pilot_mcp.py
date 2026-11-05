"""Native MCP transport plus isolated CPU execution of agent-authored robot programs."""
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import sys
import sysconfig

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ['PILOT_WORKSPACE']).resolve()
OUT = Path(os.environ['PILOT_OUTPUT']).resolve()
SOCKET = Path(os.environ['PILOT_SOCKET'])


def rpc(payload):
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(650)
        sock.connect(str(SOCKET))
        f = sock.makefile('rwb')
        f.write((json.dumps(payload)+'\n').encode()); f.flush()
        return json.loads(f.readline())


def run_code(path):
    path = (WORK/path).resolve()
    if not path.is_relative_to(WORK) or path.suffix != '.py':
        raise ValueError('Execute a Python file inside the task workspace')
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    archive = OUT/'executions'; archive.mkdir(exist_ok=True)
    (archive/(digest+'.py')).write_bytes(content)
    command = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session',
        '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib',
        '--ro-bind', '/lib64', '/lib64', '--proc', '/proc', '--dev', '/dev',
        '--tmpfs', '/tmp', '--dir', '/opt', '--dir', '/run',
        '--ro-bind', str(Path(sys.prefix)), str(Path(sys.prefix)),
        '--ro-bind', str(Path(sys.base_prefix)), str(Path(sys.base_prefix)),
        '--ro-bind', str(ROOT/'src/robot_api.py'), '/opt/robot_api.py',
        '--ro-bind' if os.environ.get('PILOT_READONLY_WORKSPACE')=='1' else '--bind',
        str(WORK), '/workspace', '--bind', str(SOCKET.parent), '/run/robot',
        '--chdir', '/workspace', '--clearenv', '--setenv', 'PATH', str(Path(sys.prefix)/'bin')+':/usr/bin:/bin',
        '--setenv', 'PYTHONPATH', '/opt', '--setenv', 'OMP_NUM_THREADS', '1',
        '--setenv', 'OPENBLAS_NUM_THREADS', '1']
    # Programs have numerical/image analysis packages, but cannot instantiate uncounted copies
    # of the benchmark from the host's installed simulator packages.
    site = Path(sysconfig.get_paths()['purelib'])
    for package in ['dm_control', 'gymnasium', 'gymnasium_robotics', 'mujoco']:
        if (site/package).exists():
            command += ['--tmpfs', str(site/package)]
    # uv's interpreter symlink may name an alias of sys.base_prefix.
    # Bind that same runtime read-only at its alias; do not expose the host home.
    executable=Path(sys.prefix)/'bin/python'
    if executable.is_symlink():
        target=Path(os.readlink(executable))
        if target.is_absolute() and target.resolve().is_relative_to(Path(sys.base_prefix).resolve()):
            alias=target.parent.parent
            if alias!=Path(sys.base_prefix):
                command+=['--ro-bind',str(Path(sys.base_prefix).resolve()),str(alias)]
    command += [str(Path(sys.executable)), '-B', '-u', '/workspace/'+str(path.relative_to(WORK))]
    started = time.time()
    try:
        done = subprocess.run(command, capture_output=True, timeout=600)
        result = dict(returncode=done.returncode, stdout=done.stdout.decode(errors='replace'),
                      stderr=done.stderr.decode(errors='replace'))
    except subprocess.TimeoutExpired as exc:
        result = dict(returncode=124, stdout=(exc.stdout or b'').decode(errors='replace'),
                      stderr=(exc.stderr or b'').decode(errors='replace')+'\nProgram wall limit: 600s')
    result.update(code_sha256=digest, seconds=time.time()-started)
    record = archive/(str(time.time_ns())+'.json')
    record.write_text(json.dumps(result))
    result['stdout'] = result['stdout'][-24000:]
    result['stderr'] = result['stderr'][-12000:]
    return result


async def main():
    server = Server('robot')

    @server.list_tools()
    async def list_tools():
        return [types.Tool(name='observe', description='Inspect current native benchmark sensors, task status, RGB and metric depth. No time advance.',
                  inputSchema={'type':'object','properties':{},'additionalProperties':False}),
                types.Tool(name='run_python_file', description='Execute a Python program you wrote in the task workspace. CPU NumPy, SciPy, image processing and file access are available. Import robot_api as robot for observe(), step(action,repeat), reset(), finish(). You can implement feedback controllers and optimize them across development episodes. Returns output, errors and current camera observations.',
                  inputSchema={'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False})]

    @server.call_tool()
    async def call(name, arguments):
        started = time.time()
        result = await asyncio.to_thread(run_code, arguments['path']) if name == 'run_python_file' else {}
        observation = await asyncio.to_thread(rpc, {'kind':'inspect'})
        images = observation.pop('images', [])
        result.update(observation)
        with (OUT/'mcp.jsonl').open('a') as log:
            log.write(json.dumps(dict(time=started, tool=name, arguments=arguments, result=result))+'\n')
        blocks = [types.TextContent(type='text', text=json.dumps(result))]
        blocks += [types.ImageContent(type='image', data=x, mimeType='image/jpeg') for x in images]
        return types.CallToolResult(content=blocks, isError=bool(result.get('error') or result.get('returncode',0)))

    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


if __name__ == '__main__':
    asyncio.run(main())
