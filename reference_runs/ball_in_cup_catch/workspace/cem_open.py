"""Open-loop CEM on my model from a settled hanging state (no dev episodes).
Question: how fast can a catch be from the hardest start, and what does the manoeuvre look like?"""
import time
import numpy as np
import vecsim

rng = np.random.default_rng(0)
HZ = -(vecsim.MC + vecsim.MB) * vecsim.G / 20.0
T, K, N, ITERS = 150, 38, 300, 70
q0 = np.array([[0.0, HZ, 0.0, HZ - 0.008]])
v0 = np.zeros((1, 4))
# settle
for _ in range(200):
    q0, v0 = vecsim.step(q0, v0, np.zeros((1, 2)))
print("settled", q0.round(4), v0.round(4))
knots_t = np.linspace(0, T - 1, K)


def expand(P):
    idx = np.arange(T)
    hi = np.clip(np.searchsorted(knots_t, idx), 1, K - 1)
    lo = hi - 1
    w = ((idx - knots_t[lo]) / (knots_t[hi] - knots_t[lo]))[None, :, None]
    return P[:, lo] * (1 - w) + P[:, hi] * w


def score(P):
    U = expand(np.clip(P, -1, 1))
    n = len(P)
    Q, V = vecsim.rollout(np.repeat(q0, n, 0), np.repeat(v0, n, 0), U)
    lx = Q[:, :, 2] - Q[:, :, 0]
    lz = (0.2 + Q[:, :, 3]) - (0.6 + Q[:, :, 1])
    inside = (np.abs(lx) < 0.025) & (np.abs(lz + 0.05) < 0.025)
    dist = np.abs(lx) + np.abs(lz + 0.05)
    return 10.0 * inside.sum(1) - 20.0 * dist.min(1) + 5.0 * np.minimum(lz.max(1), 0.1), inside, Q


mu = np.zeros((K, 2))
std = np.full((K, 2), 0.8)
t0 = time.time()
best, bestP = -1e9, None
for it in range(ITERS):
    P = np.clip(mu + std * rng.standard_normal((N, K, 2)), -1.5, 1.5)
    if bestP is not None:
        P[0] = bestP
    S, inside, Q = score(P)
    e = np.argsort(-S)[:N // 10]
    mu = 0.3 * mu + 0.7 * P[e].mean(0)
    std = 0.3 * std + 0.7 * P[e].std(0) + 0.02
    if S.max() > best:
        best, bestP = S.max(), P[int(np.argmax(S))].copy()
    if it % 10 == 0 or it == ITERS - 1:
        print(f"it {it} best {best:.2f} in-target ticks {inside[int(np.argmax(S))].sum()} t={time.time()-t0:.1f}s", flush=True)

S, inside, Q = score(bestP[None])
U = expand(np.clip(bestP[None], -1, 1))[0]
Q = Q[0]
ins = inside[0]
first = int(np.argmax(ins)) if ins.any() else None
print("first in-target tick", first, "ticks in target", int(ins.sum()), "of", T)
np.savez("memory/cem_open.npz", U=U, Q=Q)
print("tick  ux   uz   cup_x cup_z ball_x ball_z  rx rz  lz")
for k in range(0, T, 5):
    cx, cz, bx, bz = Q[k]
    rx, rz = bx - cx, (0.2 + bz) - (0.492 + cz)
    print(f"{k:3d} {U[k,0]:+.2f} {U[k,1]:+.2f}  {cx:+.3f} {cz:+.3f} {bx:+.3f} {bz:+.3f}  {rx:+.3f} {rz:+.3f}  {rz-0.108+0.0:+.3f} {'IN' if ins[k] else ''}")
