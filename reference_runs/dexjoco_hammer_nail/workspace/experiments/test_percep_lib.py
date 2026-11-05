"""Load saved observations (E1 run calls or output/calls tags) as perception.Obs."""
import json
import sys

sys.path.insert(0, 'code/hammer')
import perception as pc

RUN = 'runtime/autonomous_controller/runs/interactive-6667dfe599094cffb8d1e7a3f962ff3e'


def load_run_call(n, run=RUN):
    d = f'{run}/calls/{n:06d}'
    resp = json.load(open(f'{d}/response.json'))
    meta = json.loads(resp['content'][0]['text'])
    blobs = {}
    for a in meta.get('attachments', []):
        blobs[(a['camera'], a['kind'])] = open(f"{d}/image_{a['content_index']:02d}.png", 'rb').read()
    return pc.Obs(meta, blobs)


def load_saved(tag):
    meta = json.load(open(f'output/calls/{tag}.json'))
    blobs = {}
    for a in meta.get('attachments', []):
        blobs[(a['camera'], a['kind'])] = open(f"output/calls/{tag}_img{a['content_index']}.png", 'rb').read()
    return pc.Obs(meta, blobs)
