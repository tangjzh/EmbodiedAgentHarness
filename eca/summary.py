"""Read native outcomes without inventing thresholds for continuous tasks."""
import json
from pathlib import Path
def summarize(root):
    root=Path(root);rows=[]
    for p in root.rglob('outcome.json'):
        value=json.loads(p.read_text())
        if value.get('phase') in ['heldout','formal']:rows.append(value)
    result=dict(valid_recorded_episodes=len(rows),episodes=rows)
    binary=[r.get('native_metric',{}).get('native_final_success',r.get('native_success')) for r in rows]
    if binary and all(isinstance(v,bool) for v in binary):result['final_successes']=sum(binary)
    if rows and all('native_return' in r for r in rows):result['mean_native_return']=sum(r['native_return'] for r in rows)/len(rows)
    result['note']='Uses native fields. Continuous rewards/scores are not converted to task success. Check executor return codes and validity records before treating an episode as valid.'
    return result
