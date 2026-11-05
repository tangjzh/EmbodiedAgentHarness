import numpy as np
d = np.load('/tmp/cams0.npz')
K=np.array([[480.,0,320],[0,480,240],[0,0,1]]); T=np.array([[-0.9999999999999973, 7.118770902390902e-08, 1.5893247288229852e-08, 0.6000000238418579], [3.415060091032213e-08, 0.6494858276558781, -0.7603736973844886, 0.23999999463558197], [-6.445170038797632e-08, -0.7603736973844859, -0.649485827655879, 0.23999999463558197], [0.0, 0.0, 0.0, 1.0]])
D = d['closeup_depth_m']; H,W = D.shape
u,v = np.meshgrid(np.arange(W)+0.5, np.arange(H)+0.5)
P = np.stack([(u-K[0,2])/K[0,0]*D,(v-K[1,2])/K[1,1]*D,D,np.ones_like(D)],-1) @ T.T
rgb=d['closeup_rgb'].astype(int)
yel=(rgb[...,0]>150)&(rgb[...,1]>150)&(rgb[...,2]<80)
z=P[...,2]
m=(z>0.01)&(z<0.2)&~yel
vs,us=np.nonzero(m); print('rows',vs.min(),vs.max(),'cols',us.min(),us.max())
hist,edges=np.histogram(z[m],bins=40,range=(0.01,0.2)); 
for h,e in zip(hist,edges): print(round(e,4),h)
Y=P[yel]; print('yellow z range',Y[:,2].min(),Y[:,2].max(), 'xy mean',Y[:,:2].mean(0))
lo=Y[Y[:,2]<Y[:,2].min()+0.004]; print('peg tip est',lo[:,:3].mean(0), len(lo))
