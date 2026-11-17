"""Optional installed-runtime validation. No container start, image pull or model call."""
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eca.catalog import tasks
from eca.efe import native_command, verify_source
from harbor.models.task.task import Task
from harbor.models.job.config import JobConfig
import efe
from harbor.cli.main import app


def main():
    count=Counter()
    rows=[t for t in tasks() if t['backend']=='efe']
    for item in rows:
        for mode,spec in item['modes'].items():
            args=SimpleNamespace(command='qualify',efe_mode=mode,wall_seconds=None)
            _,path=verify_source(item,args)
            task=Task(path)
            assert task.config.agent.timeout_sec==spec['agent']['timeout_sec']
            count[mode]+=1
    # Exercise the actual CLI parser/config, while disabling its image pre-pull only
    # in this config-only check. Real runs always retain the upstream pre-pull.
    efe.prepull=lambda *a,**k: []
    for action,agent in [('develop','claude-code'),('replay','oracle'),('qualify','nop')]:
        args=SimpleNamespace(command=action,efe_mode='standard',wall_seconds=None,
                             model='claude-opus-5-5',engine='docker')
        _,path=verify_source(rows[0],args)
        command=native_command(rows[0],args,path,Path('/tmp/efe-config-only'))
        sys.argv=[command[0],*command[1:],'--print-config']
        with contextlib.redirect_stdout(io.StringIO()) as output:
            try: app()
            except SystemExit as e: assert e.code in [None,0],e
        config=JobConfig.model_validate(json.loads(output.getvalue()))
        assert config.agents[0].name==agent
        assert len(config.tasks)==1 and config.tasks[0].path==path
        assert config.n_concurrent_trials==1
    print(json.dumps(dict(task_configs=dict(count),native_cli_agents=3,
                          model_calls=0,container_starts=0)))


if __name__=='__main__': main()
