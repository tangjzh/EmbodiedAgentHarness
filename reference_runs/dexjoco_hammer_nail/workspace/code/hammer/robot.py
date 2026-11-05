"""Thin wrapper over the supplied MCP context: steps, observations, episode bookkeeping, logging."""
import json
import os
import time

import numpy as np

from motion import continuous_rotvec, rot_from_state, row
from perception import Obs

MAX_ROWS_PER_CALL = 100


class EpisodeOver(Exception):
    """The episode ended (success, step limit or failure); stop calling tools."""


class Log:
    def __init__(self, path="output/controller_log.txt"):
        self.t0 = time.time()
        self.path = path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.f = open(path, "a", buffering=1)
        except OSError:
            self.f = None

    def __call__(self, *parts, **kv):
        msg = " ".join(str(p) for p in parts)
        if kv:
            msg += " " + json.dumps(kv, default=_jsonable)
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        if self.f:
            try:
                self.f.write(line + "\n")
            except OSError:
                pass


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return np.round(o, 4).tolist()
    if isinstance(o, (np.floating,)):
        return round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    return str(o)


class Robot:
    def __init__(self, ctx, log):
        self.ctx = ctx
        self.log = log
        self.ended = False
        self.complete = False
        self.remaining = None
        self.step_id = 0
        self.state = None
        self.target = None          # last commanded row
        self.n_calls = 0
        self.feedback = None

    # ---- bookkeeping ----
    def _absorb(self, meta):
        if not isinstance(meta, dict):
            return
        if meta.get("episode_ended"):
            self.ended = True
        tr = meta.get("transition") or {}
        if tr.get("task_complete") or meta.get("task_complete"):
            self.complete = True
        if tr.get("feedback"):
            self.feedback = tr.get("feedback")
        lim = meta.get("episode_limits") or {}
        if lim.get("remaining_episode_steps") is not None:
            self.remaining = int(lim["remaining_episode_steps"])
        if meta.get("step_id") is not None:
            self.step_id = meta["step_id"]
        st = meta.get("state")
        if st:
            self.state = st

    def check(self):
        if self.ended or self.complete:
            raise EpisodeOver()

    # ---- tools ----
    def observe(self):
        self.check()
        reply = self.ctx.call("dexjoco_observe")
        self.n_calls += 1
        obs = Obs.from_reply(reply)
        self._absorb(obs.meta)
        return obs

    def step(self, rows):
        """Send rows (chunked); returns the last observation. Raises EpisodeOver when it ends."""
        self.check()
        obs = None
        i = 0
        while i < len(rows):
            chunk = rows[i:i + MAX_ROWS_PER_CALL]
            if self.remaining is not None:
                chunk = chunk[:max(0, self.remaining)]
            if not chunk:
                self.ended = True
                raise EpisodeOver()
            reply = self.ctx.call("dexjoco_step", actions=[[float(v) for v in r] for r in chunk])
            self.n_calls += 1
            obs = Obs.from_reply(reply)
            self._absorb(obs.meta)
            self.target = list(chunk[-1])
            i += len(chunk)
            if self.ended or self.complete:
                raise EpisodeOver()
        return obs

    # ---- poses ----
    def pose(self):
        st = self.state
        return np.asarray(st[0:3], float), rot_from_state(st), list(st[7:23])

    def hold_row(self):
        """Row that keeps the last commanded target, or the measured pose if none."""
        if self.target is not None:
            return list(self.target)
        p, r, h = self.pose()
        return row(p, continuous_rotvec(r, np.array([-np.pi, 0, 0])), h)

    def hold(self, n):
        return self.step([self.hold_row()] * int(n))
