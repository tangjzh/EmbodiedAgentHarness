This is the final turn, so the code below has to work in one pass. It locates the knob from the wrist camera and has fallbacks if any of that fails. It then grasps the knob's handle bar and turns it one way. If the camera sees no new red pixels (which I'm treating as a sign of flame), it turns the knob the other way. That check is only a heuristic and doesn't prove the stove is on.

```python
import numpy as np, math

def qmul(a,b):
    w1,x1,y1,z1=a; w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
def rotz(q,th):
    return qmul(np.array([math.cos(th/2),0,0,math.sin(th/2)]), np.array(q,dtype=float))
def qmat(q):
    w,x,y,z=[float(t) for t in q]; n=math.sqrt(w*w+x*x+y*y+z*z); w,x,y,z=w/n,x/n,y/n,z/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def get_cam(obs):
    cam=obs.get("robot0_eye_in_hand") if isinstance(obs,dict) else None
    items=[]
    def walk(d):
        if isinstance(d,dict):
            for k,v in d.items():
                if isinstance(v,dict): walk(v)
                elif isinstance(v,str): pass
                else: items.append((str(k).lower(),v))
        elif d is not None:
            items.append(("raw",d))
    walk(cam)
    rgb=None; depth=None; K=None
    for k,v in items:
        try:
            a=np.asarray(v)
            if a.dtype==object: continue
        except Exception:
            continue
        if a.ndim==3 and a.shape[2]==3 and a.shape[0]>32 and rgb is None:
            rgb=a.astype(float)
        el