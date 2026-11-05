"""Control-limited iLQR on the exact analytic model, plus TVLQR and closed-loop evaluation."""
import numpy as np
from scipy.linalg import solve_discrete_are

import acrobot_model as m

KAPPA2 = -2.0 * np.log(0.1)
ALPHAS = np.array([1.0, 0.6, 0.35, 0.2, 0.1, 0.05, 0.02, 0.008, 0.003, 0.001])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def jacobians(X, U):
    """Complex-step Jacobians of the RK4 step. X (T,4), U (T,). Returns A (T,4,4), B (T,4)."""
    h = 1e-30
    T = len(U)
    Xc = np.repeat(X[:, None, :], 5, axis=1).astype(complex)
    Uc = np.repeat(U[:, None], 5, axis=1).astype(complex)
    for i in range(4):
        Xc[:, i, i] += 1j * h
    Uc[:, 4] += 1j * h
    F = m.step(Xc, Uc).imag / h
    return np.transpose(F[:, :4, :], (0, 2, 1)), F[:, 4, :]


def upright_lqr(q=(10.0, 10.0, 1.0, 1.0), r=1.0):
    A, B = jacobians(np.zeros((1, 4)), np.zeros(1))
    A = A[0]
    B = B[0].reshape(4, 1)
    Q = np.diag(q)
    R = np.array([[r]])
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K[0], P


class Cost:
    """Stage cost on x_t (t=0..T) and u_t, terminal quadratic on wrapped state.

    l_t = w_rew[t]*(-reward) + w_ang[t]*((1-cos q1)+(1-cos q12)) + 0.5*w_vel[t]*|dq|^2 + 0.5*ru*u^2
    """

    def __init__(self, T, w_rew, w_ang, w_vel, ru, Pf):
        self.T = T
        self.w_rew = np.broadcast_to(np.asarray(w_rew, float), (T + 1,)).copy()
        self.w_ang = np.broadcast_to(np.asarray(w_ang, float), (T + 1,)).copy()
        self.w_vel = np.broadcast_to(np.asarray(w_vel, float), (T + 1,)).copy()
        self.ru = ru
        self.Pf = Pf

    def total(self, X, U):
        """X (...,T+1,4), U (...,T)."""
        r = m.reward(X)
        q1, q12 = X[..., 0], X[..., 0] + X[..., 1]
        ang = (1 - np.cos(q1)) + (1 - np.cos(q12))
        vel = X[..., 2] ** 2 + X[..., 3] ** 2
        stage = -self.w_rew * r + self.w_ang * ang + 0.5 * self.w_vel * vel
        out = stage.sum(axis=-1) + 0.5 * self.ru * (U ** 2).sum(axis=-1)
        xf = X[..., -1, :].copy()
        xf[..., 0] = wrap(xf[..., 0])
        xf[..., 1] = wrap(xf[..., 1])
        return out + 0.5 * np.einsum("...i,ij,...j->...", xf, self.Pf, xf)

    def derivs(self, X):
        """Returns lx (T+1,4) and Gauss-Newton style lxx (T+1,4,4)."""
        q1, q2 = X[:, 0], X[:, 1]
        q12 = q1 + q2
        s1, c1, s12, c12 = np.sin(q1), np.cos(q1), np.sin(q12), np.cos(q12)
        lx = np.zeros((self.T + 1, 4))
        lxx = np.zeros((self.T + 1, 4, 4))
        # orientation term
        lx[:, 0] += self.w_ang * (s1 + s12)
        lx[:, 1] += self.w_ang * s12
        h1 = np.maximum(c1, 0.0) * self.w_ang
        h2 = np.maximum(c12, 0.0) * self.w_ang
        lxx[:, 0, 0] += h1 + h2
        lxx[:, 0, 1] += h2
        lxx[:, 1, 0] += h2
        lxx[:, 1, 1] += h2
        # velocity term
        lx[:, 2] += self.w_vel * X[:, 2]
        lx[:, 3] += self.w_vel * X[:, 3]
        lxx[:, 2, 2] += self.w_vel
        lxx[:, 3, 3] += self.w_vel
        # reward term, zero gradient inside the 0.2 m ball
        tx = s1 + s12
        tz = c1 + c12 - 2.0
        d = np.sqrt(tx ** 2 + tz ** 2) + 1e-12
        gd1 = (tx * (c1 + c12) - tz * (s1 + s12)) / d
        gd2 = (tx * c12 - tz * s12) / d
        e = np.maximum(d - 0.2, 0.0)
        rew = np.exp(-0.5 * KAPPA2 * e ** 2)
        dcost = self.w_rew * KAPPA2 * e * rew
        lx[:, 0] += dcost * gd1
        lx[:, 1] += dcost * gd2
        curv = self.w_rew * KAPPA2 * rew * np.maximum(1.0 - KAPPA2 * e ** 2, 0.05) * (e > 0)
        lxx[:, 0, 0] += curv * gd1 * gd1
        lxx[:, 0, 1] += curv * gd1 * gd2
        lxx[:, 1, 0] += curv * gd1 * gd2
        lxx[:, 1, 1] += curv * gd2 * gd2
        # terminal quadratic
        xf = X[-1].copy()
        xf[0] = wrap(xf[0])
        xf[1] = wrap(xf[1])
        lx[-1] += self.Pf @ xf
        lxx[-1] += self.Pf
        return lx, lxx


def rollout_open(x0, U):
    X = np.zeros((len(U) + 1, 4))
    X[0] = x0
    for t, u in enumerate(U):
        X[t + 1] = m.step(X[t], u)
    return X


def ilqr(x0, U, cost, iters=100, mu=1e-3, tol=1e-6, verbose=False, deadline=None):
    """Control-limited iLQR. Returns dict with X, U, k-free TV gains, J history."""
    import time

    T = len(U)
    U = np.clip(U, -1, 1).copy()
    X = rollout_open(x0, U)
    J = float(cost.total(X, U))
    hist = [J]
    ru = cost.ru
    nA = len(ALPHAS)
    stall = 0
    for it in range(iters):
        if deadline is not None and time.time() > deadline:
            break
        A, B = jacobians(X[:-1], U)
        lx, lxx = cost.derivs(X)
        while True:
            ks = np.zeros(T)
            Ks = np.zeros((T, 4))
            Vx = lx[T].copy()
            Vxx = lxx[T].copy()
            ok = True
            for t in range(T - 1, -1, -1):
                At, Bt = A[t], B[t]
                VxxA = Vxx @ At
                VxxB = Vxx @ Bt
                Qx = lx[t] + At.T @ Vx
                Qu = ru * U[t] + Bt @ Vx
                Qxx = lxx[t] + At.T @ VxxA
                Qux = Bt @ VxxA
                Quu = ru + Bt @ VxxB
                # Tassa-style regularisation on the value Hessian for the gains only
                Quu_reg = Quu + mu * (Bt @ Bt) + 1e-9
                Qux_reg = Qux + mu * (Bt @ At)
                if Quu_reg <= 0:
                    ok = False
                    break
                k = -Qu / Quu_reg
                K = -Qux_reg / Quu_reg
                # the forward pass applies clip(U + alpha*k + K dx), the exact LQ answer under a box
                ks[t], Ks[t] = k, K
                un = U[t] + k
                if un > 1.0 or un < -1.0:
                    # saturated: value model has no response to small state changes
                    kc = (1.0 if un > 1.0 else -1.0) - U[t]
                    Vx = Qx + Qux * kc
                    Vxx = Qxx.copy()
                else:
                    Vx = Qx + K * (Quu * k + Qu) + Qux * k
                    Vxx = Qxx + Quu * np.outer(K, K) + np.outer(K, Qux) + np.outer(Qux, K)
                Vxx = 0.5 * (Vxx + Vxx.T)
            if ok:
                break
            mu = min(mu * 10, 1e8)
            if mu >= 1e8:
                return dict(X=X, U=U, J=J, hist=hist, iters=it)
        # batched closed-loop line search
        Xn = np.zeros((nA, T + 1, 4))
        Un = np.zeros((nA, T))
        Xn[:, 0] = x0
        for t in range(T):
            du = ALPHAS * ks[t] + (Xn[:, t] - X[t]) @ Ks[t]
            Un[:, t] = np.clip(U[t] + du, -1, 1)
            Xn[:, t + 1] = m.step(Xn[:, t], Un[:, t])
        Js = cost.total(Xn, Un)
        Js = np.where(np.isfinite(Js), Js, np.inf)
        best = int(np.argmin(Js))
        if Js[best] < J:
            impr = J - Js[best]
            X, U, J = Xn[best].copy(), Un[best].copy(), float(Js[best])
            hist.append(J)
            mu = max(mu * 0.5, 1e-6)
            stall = stall + 1 if impr < tol * max(1.0, abs(J)) else 0
            if verbose and it % 10 == 0:
                print(f"  it {it} J {J:.6f} alpha {ALPHAS[best]} mu {mu:.2e}")
            if stall >= 4:
                break
        else:
            mu = mu * 4
            if mu > 1e7:
                break
    return dict(X=X, U=U, J=J, hist=hist, iters=it + 1)


def tvlqr(X, U, Pf, q=(10.0, 10.0, 1.0, 1.0), r=1.0):
    """Unconstrained time-varying LQR gains along a nominal path (u = u_nom - K dx)."""
    T = len(U)
    A, B = jacobians(X[:-1], U)
    Q = np.diag(q)
    P = Pf.copy()
    Ks = np.zeros((T, 4))
    for t in range(T - 1, -1, -1):
        At, Bt = A[t], B[t]
        PB = P @ Bt
        S = r + Bt @ PB
        K = (PB @ At) / S
        Ks[t] = K
        P = Q + At.T @ P @ At - np.outer(At.T @ PB, K)
        P = 0.5 * (P + P.T)
    return Ks


def closed_loop(x0, Xn, Un, Ks, K_hold, T_total=1000):
    """Evaluate plan + TVLQR followed by saturating LQR hold. Mirrors what the policy will do."""
    x = np.array(x0, float)
    ret = 0.0
    rs = np.zeros(T_total)
    U = np.zeros(T_total)
    Xs = np.zeros((T_total + 1, 4))
    Xs[0] = x
    n = len(Un)
    for t in range(T_total):
        if t < n:
            dx = x - Xn[t]
            dx[:2] = wrap(dx[:2])
            u = Un[t] - Ks[t] @ dx
        else:
            xw = x.copy()
            xw[:2] = wrap(xw[:2])
            u = -K_hold @ xw
        u = float(np.clip(u, -1, 1))
        U[t] = u
        x = m.step(x, u)
        Xs[t + 1] = x
        rs[t] = float(m.reward(x))
        ret += rs[t]
    return dict(ret=ret, rewards=rs, U=U, X=Xs)
