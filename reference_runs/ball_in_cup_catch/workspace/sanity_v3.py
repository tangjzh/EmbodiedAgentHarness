"""Offline sanity for v3 before spending episodes: format handling, JSON-safe memory, timing, catch on both plants."""
import json
import time
import numpy as np
import controller_v3 as C
import eval_sim
import mysim

obs_d = {"position": [0.0, 0.0, 0.11, 0.33], "velocity": [0.0, 0.0, 0.0, 0.0]}
mem = {}
t0 = time.time()
a1 = C.policy(obs_d, 0, mem)
t1 = time.time()
a2 = C.policy(obs_d, 1, mem)
t2 = time.time()
print("first-tick ms", round(1000 * (t1 - t0)), "running-tick ms", round(1000 * (t2 - t1)), "action", a1)
json.dumps(mem)
print("memory JSON-serialisable: True; keys", sorted(mem))
print("formats agree on parse:", C._parse(obs_d) == C._parse({"observation": obs_d}) ==
      C._parse(np.array([0.0, 0.0, 0.11, 0.33, 0, 0, 0, 0])))
print("start inside cup -> mode:", (C.policy({"position": [0, 0, 0.0, 0.34], "velocity": [0, 0, 0, 0]}, 0, m := {}), m["mode"])[1])
pert = mysim.Params(mc=mysim.MC * 1.05, mb=mysim.MB * 0.95, tc=0.025, mu=0.7, dmin=0.85)
print("nominal  :", eval_sim.evaluate(C, n=8, seed=3, T=200), flush=True)
print("perturbed:", eval_sim.evaluate(C, n=8, seed=4, P=pert, T=200), flush=True)
print("start inside cup, ticks in target of 200:", eval_sim.run_episode(C, 0.0, 0.34, T=200))
