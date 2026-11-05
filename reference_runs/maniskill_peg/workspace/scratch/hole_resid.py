import numpy as np, json, cv2, sys
sys.path.insert(0,'/workspace')
from perception import *
from perception import cam_points
d=np.load('data/ep2_init.npz'); obs=json.load(open('data/ep2_init_obs.json'))
cam='hand_camera'
det=detect_hole(obs,d,cam)
rgb, dep, P, valid, K, E = cam_points(obs, d, cam)
n, c, t = det['n_out'], det['face_pt'], det['t']
m=-n
cam_c = -E[:, :3].T @ E[:, 3]
H, W = dep.shape
uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
d_cam = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)
d_w = d_cam @ E[:, :3]
denom = d_w @ n
ok = denom < -1e-6
lam = np.where(ok, ((c - cam_c) @ n) / np.where(ok, denom, -1), np.nan)
X = cam_c + lam[..., None] * d_w
ue = (X - c) @ t; ze = X[..., 2]
du, dz, dm = d_w @ t, d_w[..., 2], d_w @ m
dist = (P - c) @ n
L=0.0967
def predict(uc, zc, h):
    eps=1e-9; big=0.06
    inside = (np.abs(ue - uc) < h) & (np.abs(ze - zc) < h)
    su = np.where(du > eps, (uc + h - ue) / np.maximum(du, eps), np.where(du < -eps, (uc - h - ue) / np.minimum(du, -eps), np.inf))
    sz = np.where(dz > eps, (zc + h - ze) / np.maximum(dz, eps), np.where(dz < -eps, (zc - h - ze) / np.minimum(dz, -eps), np.inf))
    s = np.minimum(su, sz)
    se = 2 * L / np.maximum(dm, eps)
    dd = np.where(s < se, dm * s, big)
    return np.where(inside, np.clip(dd, 0, big), 0.0)
hb=np.array([-0.0627,0.1744,0.1261])
uct=(hb-c)@t
dobs=np.clip(-dist,0,0.06)
for name,(uc,zc,h) in {'truth':(uct,0.1261,0.0254),'free':(det_u:=((np.array([-0.0568,0.1739,0.121])-c)@t),0.121,0.020)}.items():
    pr=predict(uc,zc,h)
    roi=ok&valid&(np.abs(ue-uct)<0.06)&(ze>0.07)&(ze<det['face_ztop']-0.003)&(dist<0.002)
    r=(pr-dobs); r[~roi]=0
    print(name,'cost %.5f'%np.sum(np.minimum(r[roi]**2,0.004**2)),'agree %.3f'%np.mean(np.abs(r[roi])<0.003))
    # residual image
    img=np.clip((r*1000/10+0.5)*255,0,255).astype(np.uint8)
    img[~roi]=128
    cv2.imwrite(f'data/ep2_resid_{name}.png',cv2.resize(cv2.applyColorMap(img,cv2.COLORMAP_JET)[:110],(960,330),interpolation=cv2.INTER_NEAREST))
# print a few rows of dobs vs truth prediction in hole area
pr=predict(uct,0.1261,0.0254)
rows=np.where(det['mask'].any(1))[0]
print('mask rows',rows.min(),rows.max())
for rr in [0,3,6,10,15,20,25]:
    cols=np.where(det['mask'][rr])[0]
    if len(cols)==0: continue
    cs=cols[::max(1,len(cols)//8)]
    print(rr,'cols',cs,'\n obs',np.round(dobs[rr,cs]*1000,1),'\n prd',np.round(pr[rr,cs]*1000,1), '\n ze',np.round(ze[rr,cs],4))
