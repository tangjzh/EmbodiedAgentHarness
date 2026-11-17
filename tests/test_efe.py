import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from eca.catalog import tasks, task
from eca import efe
from eca.launch import run
from eca.summary import summarize


def args(**kw):
    values = dict(command='develop', model='claude-opus-5-5', wall_seconds=None,
                  efe_mode='standard', engine='docker', out=Path('/tmp/efe-unused'), dry_run=False)
    values.update(kw)
    return SimpleNamespace(**values)


class EFE(unittest.TestCase):
    def test_morphology_coverage_and_modes(self):
        rows = [t for t in tasks() if t['backend']=='efe']
        self.assertEqual(len(rows), 64)
        self.assertEqual(len({t['id'] for t in tasks()}), 99)
        self.assertEqual(len({t['robot'] for t in rows}), 20)
        self.assertEqual({t['morphology'] for t in rows}, {'dexterous_hand', 'quadruped',
            'humanoid', 'drone', 'aerial_manipulator', 'wheeled_vehicle', 'musculoskeletal', 'prosthetic_leg'})
        self.assertEqual(sum(len(t['modes']) for t in rows), 122)
        self.assertTrue(all('standard' in t['modes'] for t in rows))
        self.assertTrue(all(t['historical_result']['status']=='not_run_in_this_repository' for t in rows))

    def test_native_agents_and_fresh_runs(self):
        t = task('efe.dexhand.essential_cube_reorient')
        for action, agent in [('develop','claude-code'), ('replay','oracle'), ('qualify','nop')]:
            plan = efe.plan(t,args(command=action))
            cmd = plan['command']
            self.assertEqual(cmd[cmd.index('-a')+1],agent)
            self.assertEqual(cmd[cmd.index('-n')+1],'1')
            self.assertEqual(cmd[cmd.index('-k')+1],'1')
            self.assertEqual(cmd[cmd.index('-r')+1],'0')
            self.assertNotIn('--resume-trajectory',cmd)
            self.assertNotIn('--load-trajectory',cmd)
            self.assertNotIn('--extra-instruction',cmd)
            self.assertFalse(plan['official_leaderboard_submission'])
            self.assertEqual('-m' in cmd,action=='develop')

    def test_modes_preserve_upstream_budgets(self):
        for t in tasks():
            if t['backend']!='efe': continue
            for mode,spec in t['modes'].items():
                p = efe.plan(t,args(efe_mode=mode))
                self.assertEqual(p['wall_seconds'],spec['agent']['timeout_sec'])
                self.assertEqual(p['agent_budget'],spec['agent'])
                self.assertEqual(p['environment_budget'],spec['environment'])
                self.assertEqual(p['protocol'],spec['protocol'])
                self.assertIn(Path(spec['path']).name,p['command'][p['command'].index('-p')+1])

    def test_unsupported_modes_no_oracle_and_budget_override_rejected(self):
        t = task('efe.roboengineer.landing_random_pad')
        with self.assertRaisesRegex(ValueError,'no upstream'):
            efe.plan(t,args(efe_mode='privileged'))
        t = next(t for t in tasks() if t['backend']=='efe' and t['modes']['standard']['oracle']=='none')
        with self.assertRaisesRegex(ValueError,'no oracle'):
            efe.plan(t,args(command='replay'))
        with self.assertRaisesRegex(ValueError,'native timeouts'):
            efe.plan(t,args(wall_seconds=60))

    def test_new_backend_bypasses_legacy_model_proxy_and_config(self):
        t = task('efe.dexhand.essential_cube_reorient')
        with patch('eca.efe.run') as native, patch('eca.launch.configure') as old_config, \
             patch('eca.budget.ModelProxy') as proxy:
            run(t,args())
            native.assert_called_once()
            old_config.assert_not_called()
            proxy.assert_not_called()

    def test_endpoint_cannot_silently_widen_task_network(self):
        t=task('efe.dexhand.essential_cube_reorient')
        efe.check_endpoint(t,args(base_url='https://api.anthropic.com'))
        with self.assertRaisesRegex(ValueError,'outside'):
            efe.check_endpoint(t,args(base_url='https://example.invalid'))
        efe.check_endpoint(t,args(command='qualify'))  # No model endpoint needed.

    def test_task_hash_portability_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'.git').mkdir(); (root/'scripts').mkdir()
            runtime = root/'scripts/stamp.py'; runtime.write_text('original runtime')
            native = root/'tasks/hand'; native.mkdir(parents=True)
            instruction = native/'instruction.md'; instruction.write_text('original instruction')
            instruction.chmod(0o644)
            first = efe.tree_hash(native)
            instruction.chmod(0o664)
            self.assertEqual(first,efe.tree_hash(native))  # Git does not track host umask.
            instruction.chmod(0o755)
            self.assertNotEqual(first,efe.tree_hash(native))  # Executability matters.
            instruction.chmod(0o644)
            config = dict(commit='a'*40,runtime_files={'scripts/stamp.py':hashlib.sha256(runtime.read_bytes()).hexdigest()})
            t = dict(id='fixture',modes={'standard':{'path':'tasks/hand','oracle':'full','tree_sha256':first}})
            with patch.object(efe,'manifest',return_value=config),patch.object(efe,'source_root',return_value=root), \
                 patch.object(efe.subprocess,'check_output',return_value='a'*40):
                self.assertEqual(efe.verify_source(t,args()),(root,native))
                instruction.write_text('changed instruction')
                with self.assertRaisesRegex(ValueError,'task files changed'): efe.verify_source(t,args())
                instruction.write_text('original instruction'); runtime.write_text('changed runtime')
                with self.assertRaisesRegex(ValueError,'runtime changed'): efe.verify_source(t,args())

    def test_different_commit_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'.git').mkdir()
            with patch.object(efe,'source_root',return_value=root), \
                 patch.object(efe.subprocess,'check_output',return_value='0'*40):
                with self.assertRaisesRegex(ValueError,'commit mismatch'):
                    efe.verify_source(task('efe.dexhand.essential_cube_reorient'),args())

    def test_task_symlinks_are_not_followed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'outside-link').symlink_to('/etc/hosts')
            with self.assertRaisesRegex(ValueError,'symlink'): efe.tree_hash(root)

    def test_infra_errors_and_oracles_not_counted_as_agent_successes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            cases=[('oracle',{'success':1},'graded','reference_replay'),
                   ('nop',{'success':0},'graded','negative_control'),
                   ('claude-code',{'success':0,'infra_error':1},'infra_error','coding_agent'),
                   ('claude-code',None,'ungraded','coding_agent'),
                   ('claude-code',{'success':0,'sim_error':1},'graded','coding_agent')]
            for i,(agent,rewards,_,_) in enumerate(cases):
                d=root/str(i);d.mkdir()
                (d/'run.json').write_text(json.dumps({'efe_mode':'standard'}))
                (d/'result.json').write_text(json.dumps(dict(task_name='fixture',trial_name=str(i),
                    agent_info={'name':agent},verifier_result={'rewards':rewards})))
            (root/'result.json').write_text(json.dumps({'stats':{}}))  # Job aggregate excluded.
            result=summarize(root)
            self.assertEqual(result['harbor_trial_count'],5)
            self.assertEqual(result['valid_recorded_episodes'],0)
            self.assertNotIn('final_successes',result)
            for row,(_,rewards,status,role) in zip(result['harbor_trials'],cases):
                self.assertEqual(row['status'],status)
                self.assertEqual(row['role'],role)
                self.assertEqual(row['native_rewards'],rewards)

    def test_setup_uses_isolated_python_and_fixed_source(self):
        output=io.StringIO()
        with tempfile.TemporaryDirectory() as folder, patch.object(efe,'source_root',return_value=Path(folder)/'source'), \
             contextlib.redirect_stdout(output):
            efe.setup(SimpleNamespace(venv=Path(folder)/'venv',execute=False))
        text=output.getvalue()
        self.assertIn('uv venv --python 3.12',text)
        self.assertIn(efe.manifest()['commit'],text)
        self.assertIn('sparse-checkout',text)

    def test_zero_process_exit_does_not_hide_ungraded_or_infra_trial(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for rewards in [None, {'success':0,'infra_error':1}]:
                (root/'result.json').write_text(json.dumps(dict(task_name='fixture',trial_name='one',
                    agent_info={'name':'nop'},verifier_result={'rewards':rewards})))
                with self.assertRaisesRegex(RuntimeError,'not a task failure'):
                    efe.finish(root,0)
                self.assertEqual(json.loads((root/'runner_status.json').read_text())['graded_trials'],0)
            # A valid, unsuccessful robot episode is a completed experiment.
            (root/'result.json').write_text(json.dumps(dict(task_name='fixture',trial_name='one',
                agent_info={'name':'nop'},verifier_result={'rewards':{'success':0,'infra_error':0}})))
            efe.finish(root,0)


if __name__=='__main__': unittest.main()
