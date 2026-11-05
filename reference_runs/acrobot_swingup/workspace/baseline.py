"""LQR balancer and energy-pumping swing-up, evaluated in the validated simulator."""
import json
import multiprocessing as mp
import sys
import time

import numpy as np
from scipy.linalg import solve_discrete_are

import acrobot_model as m


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def linearize(x0=None, u0=0.0, eps=1e-6):
    x0 = np.zeros(4) if x0 is None else x0
    A = np.zeros((4, 4))
    for i in range(4):
        dx = np.zeros(4)
        dx[i] = eps
        A[:, i] = (m.step(x0 + dx, u0) - m.step(x0 - dx, u0)) / (2 * eps)
    B = ((m.step(x0, u0 + eps) - m.step(x0, u0 - eps)) / (2 * eps)).reshape(4, 1)
    return A, B


def lqr(q=(10.0, 10.0, 1.0, 1.0), r=1.0):
    A, B = linearize()
    Q = np.diag(q)
    R = np.array([[r]])
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def wrapped(x):
    out = x.copy()
    out[..., 0] = wrap(out[..., 0])
    out[..., 1] = wrap(out[..., 1])
    return out


def roa_probe(K, P, levels, n=4000, T=600, seed=0):
    """Fraction of states on each level set of V=x'Px that a saturating LQR stabilises."""
    rng = np.random.default_rng(seed)
    results = []
    L = np.linalg.cholesky(P)
    for c in levels:
        z = rng.normal(size=(n, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        x = np.linalg.solve(L.T, z.T).T * np.sqrt(c)
        for _ in range(T):
            u = np.clip(-(x @ K.T)[:, 0], -1, 1)
            x = m.step(x, u)
        ok = np.linalg.norm(wrapped(x), axis=1) < 1e-2
        results.append((float(c), float(ok.mean())))
    return results


def make_energy_law(p, K, P):
    ke, kp, kd, alpha, c_catch = p["ke"], p["kp"], p["kd"], p["alpha"], p["c_catch"]

    def law(x):
        xw = wrapped(x)
        v = np.einsum("...i,ij,...j->...", xw, P, xw)
        e = m.energy(x)
        q2 = xw[..., 1]
        pump = ke * (m.E_UP - e) * x[..., 3]
        shape = -kp * (q2 - alpha * np.arctan(x[..., 2])) - kd * x[..., 3]
        swing = pump + shape
        hold = -(xw @ K.T)[..., 0]
        return np.clip(np.where(v < c_catch, hold, swing), -1, 1)

    return law


def evaluate(law, x0, T=1000):
    x = x0.copy()
    ret = np.zeros(len(x0))
    first = np.full(len(x0), T, dtype=int)
    for t in range(T):
        x = m.step(x, law(x))
        r = m.reward(x)
        ret += r
        first = np.where((r >= 1.0) & (first == T), t + 1, first)
    held = m.tip_distance(x) < 0.2
    return ret, first, held


def random_starts(n, seed=0):
    rng = np.random.default_rng(seed)
    x0 = np.zeros((n, 4))
    x0[:, :2] = rng.uniform(-np.pi, np.pi, size=(n, 2))
    return x0


def _score(args):
    seed, K, P, x0 = args
    rng = np.random.default_rng(seed)
    p = {
        "ke": float(10 ** rng.uniform(-2.5, 0.5)),
        "kp": float(10 ** rng.uniform(-2, 1)),
        "kd": float(10 ** rng.uniform(-2.5, 0.5)),
        "alpha": float(rng.uniform(0, 3)),
        "c_catch": float(10 ** rng.uniform(0, 2.5)),
    }
    ret, first, held = evaluate(make_energy_law(p, K, P), x0)
    return float(ret.mean()), float(np.median(ret)), float(held.mean()), p


if __name__ == "__main__":
    t0 = time.time()
    K, P = lqr()
    print("LQR K", K, "\neig closed", np.abs(np.linalg.eigvals(linearize()[0] - linearize()[1] @ K)))
    print("open-loop eig", np.abs(np.linalg.eigvals(linearize()[0])))
    for q, r in (((10, 10, 1, 1), 1.0), ((10, 10, 1, 1), 10.0), ((1, 1, 1, 1), 1.0), ((1, 1, 0.1, 0.1), 0.1), ((100, 100, 10, 10), 1.0)):
        Ki, Pi = lqr(q, r)
        print("q", q, "r", r, "K", np.round(Ki, 2))
        print("   roa", [(round(c, 1), round(f, 3)) for c, f in roa_probe(Ki, Pi, np.max(np.linalg.eigvalsh(Pi)) * np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]))])

    x0 = random_starts(64, seed=11)
    args = [(s, K, P, x0) for s in range(1500)]
    with mp.Pool(32) as pool:
        out = pool.map(_score, args, chunksize=8)
    out.sort(key=lambda o: -o[0])
    print("elapsed", round(time.time() - t0, 1))
    for o in out[:8]:
        print("mean %.1f median %.1f held %.2f" % o[:3], {k: round(v, 3) for k, v in o[3].items()})
    x_test = random_starts(400, seed=99)
    best = out[0][3]
    ret, first, held = evaluate(make_energy_law(best, K, P), x_test)
    print("best on fresh 400 starts: mean %.1f median %.1f p10 %.1f held %.3f median-first-ball %d" % (
        ret.mean(), np.median(ret), np.percentile(ret, 10), held.mean(), np.median(first)))
    json.dump({"best": best, "K": K.tolist(), "P": P.tolist()}, open("baseline_params.json", "w"), indent=1)
