import numpy as np, json, sys
from pc import cloud
from PIL import Image
tag = sys.argv[1]
c=np.load(f'{tag}.npz'); o=json.load(open(f'{tag}_obs.json'))
x0,y0,res,N=0.54,-0.08,0.001,160
Z=np.full((N,N),-1.0); C=np.zeros((N,N,3),np.uint8)
for n in ['scene','closeup','wrist']:
    cam=o['cameras'][n]; d=c[n+'_depth_m']; rgb=c[n+'_rgb']
    P=cloud(d,np.array(cam['intrinsic_cv']),cam['cam2world_cv'])
    m=d>0; p=P[m]; col=rgb[m]
    i=((p[:,0]-x0)/res).astype(int); j=((p[:,1]-y0)/res).astype(int)
    k=(i>=0)&(i<N)&(j>=0)&(j<N)&(p[:,2]<0.08)
    for a,b,z,cc in zip(i[k],j[k],p[k,2],col[k]):
        if z>Z[a,b]: Z[a,b]=z; C[a,b]=cc
np.save(f'{tag}_Z.npy',Z)
img=np.clip((Z+0.002)/0.06*255,0,255).astype(np.uint8)
Image.fromarray(img).resize((480,480),0).save(f'/tmp/{tag}_Z.png')
Image.fromarray(C).resize((480,480),0).save(f'/tmp/{tag}_C.png')
