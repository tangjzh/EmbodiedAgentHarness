"""Task contracts; reference programs are operator-only."""
import json
import os
from pathlib import Path
ROOT=Path(os.environ.get('ECA_ROOT',Path(__file__).resolve().parents[1])).resolve()
def tasks():
    return json.loads((ROOT/'tasks/catalog.json').read_text())['tasks']
def task(name):
    for item in tasks():
        if item['id']==name:return item
    raise ValueError('Unknown task: '+name)
def enabled(item):
    if not item['enabled']:
        raise ValueError('garment_fling failed native environment qualification; no historical agent run exists. It is listed for provenance, not admitted for paid development.')
