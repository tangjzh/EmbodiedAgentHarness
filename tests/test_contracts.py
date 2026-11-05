import hashlib,json,subprocess,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from eca.catalog import ROOT,tasks,task
from eca.launch import plan
class Contracts(unittest.TestCase):
    def test_task_coverage(self):
        values=tasks();self.assertEqual(len(values),35);self.assertEqual(sum(t['enabled'] for t in values),34)
        self.assertFalse(any('robocasa' in t['benchmark'].lower() for t in values))
    def test_fresh_plan_has_no_reference(self):
        for t in tasks():
            if not t['enabled']:continue
            p=plan(t,SimpleNamespace(command='develop',model='claude-opus-5-5',out=Path('/tmp/new-run'),wall_seconds=None))
            self.assertEqual(p['initialization'],'empty_task_workspace');self.assertFalse(p['reference_visible_to_agent'])
            self.assertNotIn('reference',p)
    def test_qualification_case_is_not_a_solver(self):
        with self.assertRaises(ValueError):plan(task('garment_fling'),SimpleNamespace(command='develop',model='x',out=Path('/tmp/x'),wall_seconds=None))
    def test_reference_controllers_exist(self):
        for t in tasks():
            if t['backend'] in ['state','precision','soft','factory'] and t['enabled']:
                self.assertTrue((ROOT/t['reference']/'controller.py').is_file(),t['id'])
    def test_all_launches_plan_without_simulator_or_api(self):
        for t in tasks():
            if not t['enabled']:continue
            p=subprocess.run([sys.executable,'-B','-m','eca','develop',t['id'],'--out','/tmp/unused-eca-run','--dry-run'],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr);self.assertTrue(json.loads(p.stdout)['no_cross_task_transfer'])
    def test_reference_hashes(self):
        for row in json.loads((ROOT/'provenance/reference_manifest.json').read_text()):
            self.assertEqual(hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest(),row['sha256'])
if __name__=='__main__':unittest.main()
