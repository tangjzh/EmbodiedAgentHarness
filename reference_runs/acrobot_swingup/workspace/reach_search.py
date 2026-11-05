"""Forward reachability search over bang-bang actions with state-cell dedupe.

Each state cell keeps its earliest arrival, so the search approximates a
time-optimal path to a relaxed neighbourhood of upright. iLQR then makes the
landing exact. A child that stays in its parent's cell is allowed through,
otherwise slow states near rest would prune themselves immediately.
"""
import numpy as np

import acrobot_model as m


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def goal_score(x, q_scale=0.35, v_scale=1.5):
    qw = wrap(x[..., :2])
    return (qw ** 2).sum(axis=-1) / q_scale ** 2 + (x[..., 2:] ** 2).sum(axis=-1) / v_scale ** 2


def cell_index(x, shape, vmax, phase=(0.0, 0.0, 0.0, 0.0)):
    """phase shifts the cell grid by a fraction of a cell per dimension, so that repeated
    searches with different phases keep different representatives."""
    n1, n2, n3, n4 = shape
    i1 = np.floor((wrap(x[:, 0]) + np.pi) / (2 * np.pi) * n1 + phase[0]).astype(np.int64) % n1
    i2 = np.floor((wrap(x[:, 1]) + np.pi) / (2 * np.pi) * n2 + phase[1]).astype(np.int64) % n2
    i3 = np.clip(np.floor((x[:, 2] + vmax[0]) / (2 * vmax[0]) * n3 + phase[2]).astype(np.int64), 0, n3 - 1)
    i4 = np.clip(np.floor((x[:, 3] + vmax[1]) / (2 * vmax[1]) * n4 + phase[3]).astype(np.int64), 0, n4 - 1)
    return ((i1 * n2 + i2) * n3 + i3) * n4 + i4


def search(x0, hold=8, max_layers=110, shape=(64, 64, 40, 60), vmax=(12.0, 20.0),
           e_margin=8.0, cap=1500000, extra_layers=4, top_k=12, seed=0, goal=1.0, verbose=False, min_layer=0,
           phase=(0.0, 0.0, 0.0, 0.0)):
    """Returns (candidates, stats). Each candidate is (score, layer, action sequence at tick rate).

    min_layer delays the goal test, so a start that already looks like the goal is not
    reported as solved before the arm has had time to fall and come back around.
    """
    rng = np.random.default_rng(seed)
    seen = np.zeros(int(np.prod(shape)), dtype=bool)
    x = np.asarray(x0, float)[None, :]
    cells = cell_index(x, shape, vmax, phase)
    seen[cells] = True
    parents, acts = [], []
    found_layer = None
    cands = []
    peak = 0
    for layer in range(1, max_layers + 1):
        F = len(x)
        y = np.concatenate([x, x], axis=0)
        u = np.concatenate([np.full(F, -1.0), np.full(F, 1.0)])
        par = np.concatenate([np.arange(F), np.arange(F)])
        par_cell = cells[par]
        for _ in range(hold):
            y = m.step(y, u)
        keep = (np.abs(y[:, 2]) < vmax[0]) & (np.abs(y[:, 3]) < vmax[1]) & (m.energy(y) < m.E_UP + e_margin)
        y, u, par, par_cell = y[keep], u[keep], par[keep], par_cell[keep]
        c = cell_index(y, shape, vmax, phase)
        fresh = (~seen[c]) | (c == par_cell)
        y, u, par, c = y[fresh], u[fresh], par[fresh], c[fresh]
        if len(y) == 0:
            break
        uc, idx = np.unique(c, return_index=True)
        y, u, par, c = y[idx], u[idx], par[idx], uc
        if len(y) > cap:
            sel = rng.choice(len(y), cap, replace=False)
            y, u, par, c = y[sel], u[sel], par[sel], c[sel]
        seen[c] = True
        parents.append(par.astype(np.int32))
        acts.append((u > 0).astype(np.int8))
        x, cells = y, c
        peak = max(peak, len(x))
        s = goal_score(x)
        order = np.argsort(s)[:top_k]
        if verbose and layer % 10 == 0:
            print(f"   layer {layer} t={layer * hold * m.DT:.2f}s frontier {len(x)} best {s[order[0]]:.3f}")
        if s[order[0]] < goal and found_layer is None and layer >= min_layer:
            found_layer = layer
        if found_layer is not None:
            for j in order:
                if s[j] < goal:
                    cands.append((float(s[j]), layer, int(j)))
            if layer >= found_layer + extra_layers:
                break
    out = []
    for score, layer, j in cands:
        seq = np.zeros(layer)
        i = j
        for l in range(layer - 1, -1, -1):
            seq[l] = 1.0 if acts[l][i] else -1.0
            i = parents[l][i]
        out.append((score, layer, np.repeat(seq, hold)))
    return out, dict(found_layer=found_layer, peak=peak, layers=len(parents), hold=hold)
