import numpy as np, json, sys
d = np.load('/tmp/cams0.npz')
cal = {'closeup':(np.array([[480.,0,320],[0,480,240],[0,0,1]]), np.array([[-0.9999999999999973, 7.118770902390902e-08, 1.5893247288229852e-08, 0.6000000238418579], [3.415060091032213e-08, 0.6494858276558781, -0.7603736973844886, 0.23999999463558197], [-6.445170038797632e-08, -0.7603736973844859, -0.649485827655879, 0.23999999463558197], [0.0, 0.0, 0.0, 1.0]]))}
K,T = cal['closeup']
D = d['closeup_depth_m']; H,W = D.shape
u,v = np.meshgrid(np.arange(W)+0.5, np.arange(H)+0.5)
x=(u-K[0,2])/K[0,0]*D; y=(v-K[1,2])/K[1,1]*D
P = np.stack([x,y,D,np.ones_like(D)],-1) @ T.T
z = P[...,2]
print('z percentiles', np.percentile(z[np.isfinite(z)],[1,5,50,95,99]))
rgb=d['closeup_rgb'].astype(int)
yel=(rgb[...,0]>150)&(rgb[...,1]>150)&(rgb[...,2]<80)
print('yellow n',yel.sum(), 'pts min z', P[yel][:,2].min(), 'xy at lowest', P[yel][np.argsort(P[yel][:,2])[:20]][:,:3].mean(0))
m = (z>0.003)&(z<0.05)&~yel
print('obj n', m.sum(), P[m][:,:3].mean(0), np.percentile(P[m][:,2],[50,90,99]))
top = m & (z > np.percentile(z[m],95)-0.002)
print('top', P[top][:,:3].mean(0), top.sum())
print(D[200:240:5, 240:300:6])
print(np.round(P[200:240:5,240:300:6,2],4))
print(np.isfinite(D).mean(), D.max())
