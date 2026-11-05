"""One task per fresh coding-agent workspace."""
import argparse
import json
import os
from pathlib import Path
from .catalog import tasks,task
def main():
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
    q=subs.add_parser('list');q.add_argument('--json',action='store_true')
    q=subs.add_parser('show');q.add_argument('task')
    q=subs.add_parser('doctor');q.add_argument('task')
    q=subs.add_parser('setup');q.add_argument('--group',required=True);q.add_argument('--venv',type=Path);q.add_argument('--execute',action='store_true')
    q=subs.add_parser('summarize');q.add_argument('directory',type=Path)
    for mode in ['develop','qualify','replay']:
        q=subs.add_parser(mode);q.add_argument('task');q.add_argument('--out',required=True,type=Path)
        q.add_argument('--gpu',default='0');q.add_argument('--model',default=os.environ.get('ECA_MODEL','claude-opus-5-5'))
        q.add_argument('--base-url',default=os.environ.get('ANTHROPIC_BASE_URL','https://api.anthropic.com'))
        q.add_argument('--wall-seconds',type=int);q.add_argument('--dry-run',action='store_true')
    a=parser.parse_args()
    try:
        if a.command=='list':
            if a.json:print(json.dumps(tasks(),ensure_ascii=False,indent=2))
            else:
                for t in tasks():print(f"{t['id']:30} {t['group']:16} {'ready' if t['enabled'] else 'qualification failed'} | {t['name']}")
            return
        if a.command=='setup':
            from .setup import setup
            setup(a);return
        if a.command=='summarize':
            from .summary import summarize
            print(json.dumps(summarize(a.directory),ensure_ascii=False,indent=2));return
        t=task(a.task)
        if a.command=='show':print(json.dumps(t,ensure_ascii=False,indent=2));return
        if a.command=='doctor':
            from .doctor import inspect
            value=inspect(t);print(json.dumps(value,ensure_ascii=False,indent=2));raise SystemExit(0 if value['passed'] else 2)
        from .launch import run
        run(t,a)
    except (ValueError,RuntimeError,FileExistsError,FileNotFoundError) as error:parser.exit(2,str(error)+'\n')
