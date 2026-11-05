"""Offline smoke test: run controller.main against a fake context that replays saved E2 replies.

The fake picks a saved reply by the commanded phase (hand closure, flange height, single-row strikes),
keeps step_id / remaining steps consistent and ends the episode with task_complete after N strike rows.
It checks code paths (no exceptions, sane transitions), not physics.

usage: smoke_fake.py [n_strike_rows_to_complete] [--no-complete] [--lift-fail]
"""
import copy
import json
import os
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.insert(0, 'code/hammer')
import controller as C

RUN = 'runtime/autonomous_controller/runs/interactive-714857d9a56549f88ea612b3c8ce8b1c'
CACHE = {}


def saved(n):
    if n not in CACHE:
        CACHE[n] = json.load(open(f'{RUN}/calls/{n:06d}/response.json'))
    return copy.deepcopy(CACHE[n])


class FakeCtx:
    def __init__(self, n_complete=8, complete=True, lift_fail=False, limit=1000):
        self.step = 0
        self.limit = limit
        self.strike_rows = 0
        self.n_complete = n_complete
        self.complete = complete
        self.lift_fail = lift_fail
        self.calls = []

    def reply(self, n, tc=False):
        r = saved(n)
        meta = json.loads(r['content'][0]['text'])
        meta['step_id'] = self.step
        meta['episode_limits'] = {'remaining_episode_steps': self.limit - self.step}
        meta['episode_ended'] = tc or self.step >= self.limit
        meta.setdefault('transition', {})
        if isinstance(meta['transition'], dict):
            meta['transition']['task_complete'] = tc
        r['content'][0]['text'] = json.dumps(meta)
        return r

    def call(self, tool, **kw):
        self.calls.append((tool, len(kw.get('actions', []))))
        if tool == 'dexjoco_observe':
            return self.reply(1 if self.step == 0 else 9)
        if tool == 'dexjoco_hold':
            self.step += int(kw.get('steps', 1))
            return self.reply(9)
        assert tool == 'dexjoco_step', tool
        rows = kw['actions']
        assert 1 <= len(rows) <= 100, len(rows)
        for r in rows:
            assert len(r) == 22, len(r)
            assert all(v == v for v in r), 'nan in row'
        self.step += len(rows)
        last = rows[-1]
        z = last[2]
        closed = max(last[6:18]) > 1.3
        half = max(last[6:18]) > 0.3
        if len(rows) == 1 and closed and z < 1.2:
            self.strike_rows += 1
            if self.complete and self.strike_rows >= self.n_complete:
                return self.reply(30, tc=True)
            return self.reply(min(22 + self.strike_rows, 29))
        if not half:
            return self.reply(5 if z > 1.2 else 11)
        if not closed:
            return self.reply(12)
        if z < 1.15:
            return self.reply(13)
        if z < 1.3:
            return self.reply(9 if self.lift_fail else 15)   # 9: hammer still on the table, hand open
        return self.reply(20)


args = [a for a in sys.argv[1:] if not a.startswith('--')]
ctx = FakeCtx(n_complete=int(args[0]) if args else 8, complete='--no-complete' not in sys.argv,
              lift_fail='--lift-fail' in sys.argv)
c = C.main(ctx)
print('steps used', ctx.step, 'calls', len(ctx.calls))
print('summary', json.dumps(c.stats, default=str)[:1500])
