"""Operator configuration; credentials and machine paths are not bundled."""
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
RAX=Path(os.environ.get('ECA_RAX_ROOT',ROOT/'external/rax-robodojo')).resolve()
CLAUDE=Path(os.environ.get('ECA_CLAUDE_BIN') or shutil.which('claude') or 'claude')
MODEL=os.environ.get('ECA_MODEL','claude-opus-5-5')
EFFORT=os.environ.get('ECA_EFFORT','max')
IMAGE=os.environ.get('ECA_AGENT_IMAGE','eca-agent:2026-11')
