"""Read native outcomes without inventing thresholds for continuous tasks."""
import json
from pathlib import Path


def harbor_trials(root):
    """Keep every native reward and agent role; never pool reference and model results."""
    rows = []
    for path in sorted(root.rglob('result.json')):
        value = json.loads(path.read_text())
        if 'trial_name' not in value or 'task_name' not in value:
            continue  # Harbor's job-level aggregate is not another episode.
        metadata = {}
        for parent in path.parents:
            if (parent/'run.json').is_file():
                metadata = json.loads((parent/'run.json').read_text()); break
            if parent == root:
                break
        rewards = (value.get('verifier_result') or {}).get('rewards')
        exception = value.get('exception_info')
        infra = isinstance(rewards, dict) and rewards.get('infra_error') == 1
        status = ('infra_error' if infra else 'ungraded' if not rewards else
                  'exception_with_verdict' if exception else 'graded')
        agent = (value.get('agent_info') or {}).get('name', metadata.get('agent'))
        role = {'oracle': 'reference_replay', 'nop': 'negative_control'}.get(agent, 'coding_agent')
        usage = value.get('agent_result') or {}
        rows.append(dict(task=value['task_name'], trial=value['trial_name'], agent=agent,
            role=role, efe_mode=metadata.get('efe_mode'), status=status,
            native_rewards=rewards, exception_type=exception.get('exception_type') if exception else None,
            started_at=value.get('started_at'), finished_at=value.get('finished_at'),
            cost_usd=usage.get('cost_usd'), n_input_tokens=usage.get('n_input_tokens'),
            n_output_tokens=usage.get('n_output_tokens'), result_path=str(path),
            videos=[str(p) for p in sorted(path.parent.rglob('*.mp4'))]))
    return rows


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
    native=harbor_trials(root)
    if native:
        result['harbor_trials']=native
        result['harbor_trial_count']=len(native)
        result['harbor_note']='Native rewards are kept per trial, mode and agent role. Oracle and nop are not coding-agent scores; infra_error and ungraded trials are not task failures. No cross-task reward average is computed.'
    return result
