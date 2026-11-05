"""Admission to held-out robot trials is independent of CLI subtype labels."""
import json
from pathlib import Path


def agent_outcome(log, returncode, development_steps):
    events=[]
    for line in Path(log).read_text(errors='replace').splitlines():
        try:events.append(json.loads(line))
        except ValueError:pass
    result=next((x for x in reversed(events) if x.get('type')=='result'),None)
    message=str((result or {}).get('result',''))
    if returncode != 0 or result is None or result.get('is_error'):
        infrastructure=('API Error' in message or 'infrastructure' in message.lower()
                        or 'provider' in message.lower())
        return dict(eligible=False,status='infrastructure_interrupted' if infrastructure else 'agent_error',
                    reason=message[:1200] or f'CLI returncode={returncode}; terminal SDK result present={result is not None}',
                    cli_returncode=returncode,development_steps=development_steps)
    if development_steps <= 0:
        return dict(eligible=False,status='no_robot_execution',
                    reason='No native control action was executed during development',development_steps=0)
    return dict(eligible=True,status='controller_validation_pending',reason=None,
                cli_returncode=returncode,development_steps=development_steps)
