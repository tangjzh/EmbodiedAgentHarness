"""Pinned RAX controller lifecycle with no coding agent or model credential."""
import argparse
import json
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path);p.add_argument('--task');p.add_argument('--qualify',action='store_true');a=p.parse_args()
    from services.controller.frontend import load_configuration
    from services.controller.backend import make_backend
    from services.controller.supervisor import Supervisor
    config=load_configuration(a.config);a.out.mkdir(parents=True,exist_ok=False)
    if getattr(config,'benchmark','dexjoco')=='dexjoco':
        from rax_runtime_compat import UvBundleSandbox,spawn_uv_worker
        supervisor=Supervisor(config,lambda:make_backend(config),runner_factory=UvBundleSandbox,worker_factory=spawn_uv_worker)
    else:
        supervisor=Supervisor(config,lambda:make_backend(config))
    try:
        if a.qualify:
            backend=make_backend(config);backend.close()
            (a.out/'qualification.json').write_text(json.dumps(dict(passed=True,new_model_calls=0,scope='Native backend construction only; visual/action checks still required before a scored run.')));return
        relative={'insert_key':'code/insert_key','sweep_blocks':'code/sweep','dexjoco_hammer_nail':'code/hammer'}[a.task]
        source=a.reference/relative
        if not (source/'controller.py').is_file():raise RuntimeError('Complete historical bundle missing: '+str(source))
        bundle,manifest=supervisor.register(source,workspace_path=relative)
        (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        rehearsal=supervisor.run(bundle,formal=False)
        (a.out/'rehearsal.json').write_text(json.dumps(rehearsal,indent=2,default=str))
        submission=supervisor.submit(bundle)
        (a.out/'submission.json').write_text(json.dumps(submission,indent=2,default=str))
        result=supervisor.run_submission()
        (a.out/'formal_result.json').write_text(json.dumps(result,indent=2,default=str))
    finally:supervisor.close()
if __name__=='__main__':main()
