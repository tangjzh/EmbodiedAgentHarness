"""Validate mysim against a logged real episode: one-tick prediction errors + open-loop replay. No robot steps."""
import sys
import time
import numpy as np
import mysim

path = sys.argv[1] if len(sys.argv) > 1 else "memory/exp01.npz"
L = np.load(path)["log"]
U, Q, V = L[:, :2], L[:, 2:6], L[:, 6:10]
print("MC", mysim.MC, "MB", mysim.MB, "ticks", len(L))


def str_len(q):
    return np.hypot(q[..., 2] - q[..., 0], (0.2 + q[..., 3]) - (0.492 + q[..., 1]))


# one-tick prediction: log row i holds action i and the state AFTER it, so predict i+1 from state i with action i+1
errq, errv = [], []
for i in range(len(L) - 1):
    s = mysim.Sim()
    s.q, s.v = list(Q[i]), list(V[i])
    s.step(U[i + 1])
    errq.append(np.array(s.q) - Q[i + 1])
    errv.append(np.array(s.v) - V[i + 1])
errq, errv = np.array(errq), np.array(errv)
taut = str_len(Q[:-1]) > 0.3
print("one-tick RMS vel err [cx cz bx bz] all :", np.sqrt((errv ** 2).mean(0)).round(4))
print("one-tick RMS vel err taut  (n=%d)      :" % taut.sum(), np.sqrt((errv[taut] ** 2).mean(0)).round(4))
print("one-tick RMS vel err slack (n=%d)      :" % (~taut).sum(), np.sqrt((errv[~taut] ** 2).mean(0)).round(4))
print("one-tick MAX vel err                   :", np.abs(errv).max(0).round(3))
print("one-tick RMS pos err                   :", np.sqrt((errq ** 2).mean(0)).round(5))
worst = np.argsort(-np.abs(errv).max(1))[:8]
print("worst ticks:", worst.tolist(), "string", str_len(Q[worst]).round(3).tolist())

# open-loop replay from the first logged state
t0 = time.time()
s = mysim.Sim()
s.q, s.v = list(Q[0]), list(V[0])
sim_q = [list(Q[0])]
for i in range(1, len(L)):
    s.step(U[i])
    sim_q.append(list(s.q))
sim_q = np.array(sim_q)
print("replay wall s:", round(time.time() - t0, 3))
for T in (25, 50, 100, 150, 200, 300, 500, 999):
    if T < len(L):
        print(f"replay t={T/50:5.1f}s real {Q[T].round(3)} sim {sim_q[T].round(3)}  |err| {np.abs(sim_q[T]-Q[T]).max():.3f}")
print("static string length, last 100 ticks: real", str_len(Q[-100:]).mean().round(5), "sim", str_len(sim_q[-100:]).mean().round(5))
print("peak string length: real", str_len(Q).max().round(4), "sim", str_len(sim_q).max().round(4))
