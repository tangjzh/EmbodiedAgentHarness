import numpy as np, json, sys
sys.path.insert(0,'/workspace')
from perception import *
for f in ['ep3_init','ep3_s23']:
    d=np.load(f'data/{f}.npz'); obs=json.load(open(f'data/{f}_obs.json'))
    for cam in ['base_camera','hand_camera']:
        rgb, dep, P, valid, K, E = cam_points(obs, d, cam)
        orange, white, box = color_masks(rgb)
        bm = box & valid & (P[...,2] > 0.004)
        print(f,cam,'box px',bm.sum())
        det=detect_hole(obs,d,cam)
        if det is None: print('  det None'); continue
        print('  n_out',np.round(det['n_out'],3),'yaw %.1f'%(np.degrees(np.arctan2(det['n_out'][1],det['n_out'][0]))%360),'face_pt',np.round(det['face_pt'],4),'face_u',np.round(det['face_u'],4),'ztop %.4f'%det['face_ztop'],'nbehind',det.get('nbehind'),'npx',det.get('npx'))
        if det.get('hole') is not None:
            print('  coarse',np.round(det['hole'],4),det['touches'],'u_ext',np.round(det['u_ext'],4),'z_ext',np.round(det['z_ext'],4))
            for L in [0.0956]:
                rf=refine_hole(obs,d,det,cam,L=L)
                if rf is None: print('  refine None'); continue
                print('  refined',np.round(rf['hole'],4),'h %.4f agree %.3f npx %d'%(rf['h'],rf['agree'],rf['npx']))
