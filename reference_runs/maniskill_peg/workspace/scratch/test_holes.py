import numpy as np, json, time, sys
sys.path.insert(0,'/workspace')
from perception import detect_hole, refine_hole
frames={'ep1':['ep1_init','ep1_d_v0','ep1_d_v1','ep1_d_v2','ep1_d_v3','ep1_d_v4','ep1_d_v5','ep1_s20','ep1_pregrasp','ep1_grasped'],
        'ep2':['ep2_init','ep2_s18']}
Ls={'ep1':0.1117,'ep2':0.0967}
for ep,fl in frames.items():
    for f in fl:
        try:
            d=np.load(f'data/{f}.npz'); obs=json.load(open(f'data/{f}_obs.json'))
        except Exception as e:
            continue
        for cam in ['hand_camera','base_camera']:
            try:
                det=detect_hole(obs,d,cam)
            except Exception as e:
                print(f,cam,'det err',e); continue
            if det is None or det.get('hole') is None:
                print(f,cam,'no det'); continue
            t0=time.time()
            rf=refine_hole(obs,d,det,cam,h0=None,L=Ls[ep])
            dt=time.time()-t0
            tch=[k for k,v in det['touches'].items() if v]
            if rf is None:
                print(f,cam,'refine None'); continue
            print('%-13s %-11s %s h %.4f agree %.3f npx %4d touch %-12s %.2fs'%(f,cam[:4],np.round(rf['hole'],4),rf['h'],rf['agree'],rf['npx'],','.join(tch),dt))
