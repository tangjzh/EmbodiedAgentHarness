import contextlib,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from eca import setup as module
class Setup(unittest.TestCase):
    def test_existing_empty_submodule_directory_is_cloned(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'provenance').mkdir()
            (root/'provenance/upstreams.json').write_text(json.dumps({'rax-robodojo':{'url':'https://example.invalid/rax.git','commit':'a'*40},'maniskill':{'url':'https://example.invalid/mani.git','commit':'b'*40}}))
            (root/'external/rax-robodojo').mkdir(parents=True)
            output=io.StringIO()
            with patch.object(module,'ROOT',root),contextlib.redirect_stdout(output):
                module.setup(SimpleNamespace(group='maniskill',venv=None,execute=False))
            self.assertIn('git clone --no-checkout https://example.invalid/rax.git',output.getvalue())
    def test_existing_non_git_content_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'provenance').mkdir()
            (root/'provenance/upstreams.json').write_text(json.dumps({'rax-robodojo':{'url':'https://example.invalid/rax.git','commit':'a'*40}}))
            path=root/'external/rax-robodojo';path.mkdir(parents=True);(path/'keep.txt').write_text('original')
            with patch.object(module,'ROOT',root):
                with self.assertRaises(ValueError):module.setup(SimpleNamespace(group='maniskill',venv=None,execute=False))
            self.assertEqual((path/'keep.txt').read_text(),'original')
