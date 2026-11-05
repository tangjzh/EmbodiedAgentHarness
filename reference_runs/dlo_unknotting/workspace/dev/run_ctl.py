"""Dev runner: execute controller.main() for a full episode with captures. Plan/tag from dev/ctl_run.json:
{"tag": str, "capture_every": int, "plan": [[n, [P vel], [Q vel]], ...] (optional; default controller.PLAN),
 "d_max", "p_z_floor", "p_rot": [rx, ry, rz] (optional controller.main kwargs)}"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/workspace")
import controller  # noqa: E402
import robot_api as robot  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(HERE, "ctl_run.json")))
tag = cfg["tag"]
every = int(cfg.get("capture_every", 0))
plan = cfg.get("plan")


def capture(name):
    out = os.path.join(HERE, "out", tag, name)
    os.makedirs(out, exist_ok=True)
    obs = robot.observe()
    cams = robot.cameras()
    json.dump(obs, open(os.path.join(out, "obs.json"), "w"), indent=1)
    for k, v in cams.items():
        np.save(os.path.join(out, k + ".npy"), v)
        if k.endswith("_rgb"):
            cv2.imwrite(os.path.join(out, k + ".png"), cv2.cvtColor(v, cv2.COLOR_RGB2BGR))


log = []


def on_step(k, res):
    o = res.get("observation", {})
    ent = dict(k=k, step=o.get("native_step"), R=np.round(o["robot"]["right"]["eef_position_world"], 4).tolist(),
               L=np.round(o["robot"]["left"]["eef_position_world"], 4).tolist(), terminal=res.get("terminal"),
               qR=np.round(o["robot"]["right"].get("eef_quaternion_wxyz", []), 5).tolist(),
               qL=np.round(o["robot"]["left"].get("eef_quaternion_wxyz", []), 5).tolist())
    log.append(ent)
    if k % 5 == 0 or res.get("terminal"):
        print(json.dumps(ent))
    if every and k % every == 0 and not res.get("terminal"):
        capture("s%03d" % o.get("native_step"))


capture("s000")
kw = {k: cfg[k] for k in ("d_max", "p_z_floor", "p_rot") if k in cfg}
res = controller.main(plan=plan, on_step=on_step, log_path="/tmp/dev_ctl_%s.jsonl" % tag, **kw)
capture("final")
json.dump(log, open(os.path.join(HERE, "out", tag, "log.json"), "w"))
print("TERMINAL", res.get("terminal"), json.dumps(res.get("final_native_metric")))
