"""Generic runner: runs ctrl2.act for N ticks (or until terminal) from current state, logs, prints summary.
Config from run_cfg.json: {"N":..., "P":{overrides}, "tag":..., "reset":bool, "print_every":k}"""
import numpy as np, robot_api as robot, json, os, time
import controller as C
np.set_printoptions(precision=2, suppress=True, linewidth=200)
cfg = json.load(open('run_cfg.json'))
P = dict(C.DEFAULT); P.update(cfg.get('P', {}))
if cfg.get('reset'):
    r0 = robot.reset()
    print('RESET ->', {k: r0[k] for k in r0 if k != 'observation'} if isinstance(r0, dict) else r0)
o = robot.observe()
obs = o['observation'] if 'observation' in o else o
ret0 = o.get('native_return', 0.0); step0 = o.get('step')
print('start step', step0, 'ret', ret0, 'episode', o.get('episode'), 'obs', obs)
mem = {}; rows = []; t0 = time.time(); r = o
for i in range(cfg['N']):
    a, info = C.act(obs, mem, P)
    r = robot.step([float(a[0]), float(a[1])], 1)
    obs = r['observation']
    q, dq, th, w = C.decode(obs)
    rew = r['rewards'][-1] if r.get('rewards') else np.nan
    rows.append([r['step'], q[0], q[1], dq[0], dq[1], th, w, obs['touch'][0], obs['touch'][1], a[0], a[1],
                 rew, info['psi'], info['p'], C.MODES[info['mode']], info['s'], info['s_end']])
    if r.get('terminal'):
        break
arr = np.array(rows)
os.makedirs('logs', exist_ok=True)
np.save(f"logs/{cfg.get('tag', 'run')}.npy", arr)
pe = cfg.get('print_every', 1); n_print = cfg.get('n_print', 120); p0 = cfg.get('print_from', 0)
print('cols: step q1 q2 dq1 dq2 th w t0 t1 a0 a1 rew psi p mode s s_end   (psi,p,s at decision time)')
for row in arr[p0:p0 + n_print * pe:pe]:
    print(row)
w = arr[:, 6]
if cfg.get('seg'):
    # compact trace of ticks around fast-regime activity: step mode psi(dec) p(dec) w(after) touch rew
    idx = [i for i in range(len(arr)) if arr[i, 14] in (0, 1, 6) or (i > 0 and arr[i - 1, 14] in (0, 1, 6))]
    lines = [f"{int(arr[i,0])}:{'DUPpsSWw'[int(arr[i,14])] if int(arr[i,14])<8 else '?'} {arr[i,12]:5.0f} {arr[i,13]:4.0f} {arr[i,6]:6.1f} {arr[i,7]:3.1f}"
             for i in idx[:cfg.get('seg_n', 150)]]
    for k in range(0, len(lines), 5):
        print(' | '.join(lines[k:k + 5]))
print(f"ticks {len(arr)} return now {r.get('native_return')} gained {r.get('native_return', 0) - ret0} "
      f"mean w {w.mean():.2f} frac(w<=-15) {(w <= -15).mean():.3f} min w {w.min():.1f} terminal {r.get('terminal')} "
      f"secs {time.time() - t0:.1f}")
# per-100 tick summary
for k in range(0, len(arr), 100):
    seg = arr[k:k + 100]
    print(f"  ticks {int(seg[0, 0])}-{int(seg[-1, 0])}: mean w {seg[:, 6].mean():6.2f}  rew {np.nansum(seg[:, 11]):4.0f}  "
          f"modes " + ' '.join(f"{m}:{np.mean(seg[:, 14] == v):.2f}" for m, v in C.MODES.items()))
