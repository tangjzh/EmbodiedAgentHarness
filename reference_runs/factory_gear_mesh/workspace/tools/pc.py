import numpy as np, json
def cloud(d, K, T, off=0.5):
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W)+off, np.arange(H)+off)
    z = d
    x = (u-K[0,2])/K[0,0]*z; y = (v-K[1,2])/K[1,1]*z
    P = np.stack([x,y,z,np.ones_like(z)],-1) @ np.asarray(T).T
    return P[...,:3]
