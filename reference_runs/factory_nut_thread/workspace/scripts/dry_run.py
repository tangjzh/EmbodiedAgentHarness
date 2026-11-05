import json, sys, time
sys.path.insert(0, '/workspace/scripts')
import mock_robot
sys.modules['robot_api'] = mock_robot
sys.path.insert(0, '/workspace')
import controller as C
t0 = time.time()
cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
cfg.setdefault('log_dir', '/tmp/dry')
out = C.main(cfg)
print('done in %.1fs' % (time.time() - t0), out['native_step'], out['terminal'], out['native_success'])
L = [json.loads(l) for l in open(cfg['log_dir'] + '/log.jsonl')]
P = L[0]
print('perception: bolt', P['bolt']['xy'], P['bolt']['ztip'], 'nut', P['nut']['o_hand'], P['nut']['src'], [ (e['cam'], e['widths'], e['good']) for e in P['nut']['ests']])
last = None
for l in L[1:]:
    if l['kind'] == 'step' and (l['phase'] != last or l['step'] % 25 == 0):
        print(l['step'], l['phase'], 'h %.2f mm' % (l['h'] * 1e3), 'exy %.2f mm' % (l['exy'] * 1e3), 's', [round(x, 2) for x in l['s'][:3]], 'succ', l['succ'])
        last = l['phase']
