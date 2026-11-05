"""Experiment harness helpers: robust parsing of robot.step returns + logging."""
import json, os
import numpy as np
import robot_api as robot

_printed = {'done': False}


def parse(r):
    if not _printed['done']:
        _printed['done'] = True
        if isinstance(r, dict):
            print('STEP RETURN KEYS:', list(r.keys()))
            print('STEP RETURN SAMPLE:', {k: (v if not isinstance(v, list) or len(v) < 20 else str(v)[:120]) for k, v in r.items()})
        else:
            print('STEP RETURN TYPE:', type(r), str(r)[:300])
    if isinstance(r, dict):
        obs = r.get('observation', r.get('obs'))
        rew = r.get('reward', r.get('rewards', 0.0))
        if isinstance(rew, (list, tuple)):
            rew = float(np.sum(rew))
        term = bool(r.get('terminal', r.get('terminated', False)))
        trunc = bool(r.get('truncated', False))
        return np.asarray(obs, float), float(rew if rew is not None else 0.0), term, trunc, r
    if isinstance(r, (list, tuple)) and len(r) >= 3:
        return np.asarray(r[0], float), float(r[1]), bool(r[2]), False, r
    raise RuntimeError('unknown step return format')


def save_log(name, log):
    os.makedirs('logs', exist_ok=True)
    arr = {k: np.asarray(v) for k, v in log.items()}
    np.savez(os.path.join('logs', name + '.npz'), **arr)


LOGKEYS = ['obs', 'act', 'rew', 'mode', 'foot_h', 'heel_h', 'toe_h', 'xc', 'zc', 'vxc', 'vzc', 'a1', 'a2', 'a3']


def new_log():
    return {k: [] for k in LOGKEYS}


def append(log, obs, a, rew, mode, st):
    log['obs'].append(np.asarray(obs, float)); log['act'].append(np.asarray(a, float)); log['rew'].append(rew)
    log['mode'].append(mode)
    for k in LOGKEYS[4:]:
        log[k].append(float(st[k]))


def segments(log):
    """Summarize contiguous mode segments."""
    O = np.array(log['obs']); modes = log['mode']
    out = []
    s = 0
    for t in range(1, len(modes) + 1):
        if t == len(modes) or modes[t] != modes[s]:
            seg = slice(s, t)
            out.append((s, modes[s], t - s, O[seg, 1].min(), O[seg, 1].max(), O[seg, 5].mean(), O[seg, 0].max(),
                        np.array(log['foot_h'][s:t]).max()))
            s = t
    return out


def print_segments(log, limit=80):
    for (s, m, n, pmin, pmax, vx, zmax, fh) in segments(log)[:limit]:
        print(f'  t{s:4d} {m:7s} n{n:3d} p[{pmin:+.3f},{pmax:+.3f}] vx{vx:+.2f} zmax{zmax:.3f} footmax{fh:.3f}')
