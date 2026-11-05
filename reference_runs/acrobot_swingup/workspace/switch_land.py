"""Landing by Levenberg-Marquardt on bang-bang switch times.

A bang-bang signal is described by its initial sign and switch times tau (in ticks).
The control on tick t is the average of the signal over [t, t+1], so a fractional
switch time is a continuous parameter. Residual e = L' x_T with P = L L', so that
|e|^2 is the LQR cost-to-go at the end of the swing.
"""
import numpy as np

import acrobot_model as m
import traj_opt as to


def seq_to_switches(seq):
    seq = np.sign(np.where(np.asarray(seq) == 0, 1.0, seq))
    idx = np.nonzero(np.diff(seq))[0] + 1
    return float(seq[0]), idx.astype(float)


def switches_to_controls(s0, tau, T):
    """tau (..., N). Returns u (..., T)."""
    tau = np.asarray(tau, float)
    t = np.arange(T, dtype=float)
    frac = np.clip(t + 1 - tau[..., None], 0.0, 1.0)  # (..., N, T)
    signs = (-1.0) ** (np.arange(tau.shape[-1]) + 1)
    # signs apply along the switch axis
    total = np.einsum("...ij,i->...j", frac, signs) if tau.shape[-1] else np.zeros(tau.shape[:-1] + (T,))
    return np.clip(s0 * (1.0 + 2.0 * total), -1.0, 1.0)


def rollout_batch(x0, U):
    """U (B, T). Returns X (B, T+1, 4)."""
    B, T = U.shape
    X = np.zeros((B, T + 1, 4))
    X[:, 0] = x0
    for t in range(T):
        X[:, t + 1] = m.step(X[:, t], U[:, t])
    return X


def project_order(tau, T):
    """Keep switch times inside [0, T] and non-decreasing, on the last axis."""
    tau = np.clip(tau, 0.0, float(T))
    return np.maximum.accumulate(tau, axis=-1)


def residual(xf, L):
    xw = xf.copy()
    xw[..., 0] = to.wrap(xw[..., 0])
    xw[..., 1] = to.wrap(xw[..., 1])
    return xw @ L


def land_switches(x0, seq, P, iters=80, target=1.0, verbose=False):
    """Returns dict(U, X, V, tau, iters, success) with V = x_T' P x_T."""
    T = len(seq)
    s0, tau = seq_to_switches(seq)
    if len(tau) == 0:
        tau = np.array([T / 2.0, T / 2.0])
    return land_switches_tau(x0, s0, tau, T, P, iters=iters, target=target, verbose=verbose)


def land_switches_tau(x0, s0, tau, T, P, iters=80, target=1.0, verbose=False):
    """LM on switch times, starting from explicit switch times tau over a horizon of T ticks."""
    tau = np.asarray(tau, float).copy()
    L = np.linalg.cholesky(P)
    U = switches_to_controls(s0, tau, T)
    X = to.rollout_open(x0, U)
    e = residual(X[-1], L)
    V = float(e @ e)
    N = len(tau)
    signs = s0 * 2.0 * (-1.0) ** (np.arange(N) + 1)
    lams = np.array([1e-6, 1e-4, 1e-2, 1.0])
    scales = np.array([1.0, 0.5, 0.2, 0.05])
    hist = [V]
    for it in range(iters):
        if V < target:
            break
        A, B = to.jacobians(X[:-1], U)
        S = np.zeros((T, 4))
        Phi = np.eye(4)
        for t in range(T - 1, -1, -1):
            S[t] = Phi @ B[t]
            Phi = Phi @ A[t]
        ticks = np.clip(np.floor(tau).astype(int), 0, T - 1)
        # raising tau_i delays that switch; u on its tick falls by the signed jump
        J = -(S[ticks] * signs[:, None]) @ L  # (N, 4)
        J = J.T  # (4, N)
        norms = np.linalg.norm(J, axis=0) + 1e-9
        cands = []
        for variant in (0, 1):
            # variant 0 is plain minimum norm, variant 1 equalises per-switch sensitivity
            D = np.ones(N) if variant == 0 else 1.0 / norms
            JD = J * D
            G = JD @ JD.T
            gnorm = np.trace(G) / 4.0 + 1e-12
            for lam in lams:
                z = np.linalg.solve(G + lam * gnorm * np.eye(4), -e)
                step = D * (JD.T @ z)
                peak = np.abs(step).max() + 1e-12
                step = step * min(1.0, 3.0 / peak)
                for sc in scales:
                    cands.append(tau + sc * step)
        cands = project_order(np.array(cands), T)
        Uc = switches_to_controls(s0, cands, T)
        Xc = rollout_batch(x0, Uc)
        ec = residual(Xc[:, -1], L)
        Vc = (ec ** 2).sum(axis=1)
        Vc = np.where(np.isfinite(Vc), Vc, np.inf)
        j = int(np.argmin(Vc))
        if Vc[j] >= V:
            hist.append(V)
            break
        tau, U, X, e, V = cands[j], Uc[j], Xc[j], ec[j], float(Vc[j])
        hist.append(V)
        if verbose:
            print(f"   lm {it} V {V:.4g}")
    return dict(U=U, X=X, V=V, tau=tau, s0=s0, iters=len(hist) - 1, success=V < target, hist=hist)
