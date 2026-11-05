import numpy as np
from scipy.spatial.transform import Rotation as R

def q2R(q):
    return R.from_quat([q[1], q[2], q[3], q[0]])

flange_q = [4.36497393134852e-11, -0.999996875001627, 1.7459859336099726e-08, -0.0024999973958342954]
flange_p = np.array([-0.44294797221632903, 0, 1.5170134972986655])
cam_q = [0.6117026911951683, 0.3542081838182911, -0.35520854702233023, -0.611705213499951]
cam_p = np.array([-0.5827462220057252, 0.03000000488877585, 1.4763140002195247])
Rf = q2R(flange_q)
Rc = q2R(cam_q)
print('flange axes (cols x,y,z) in world:\n', Rf.as_matrix().round(4))
print('camera view dir (-Z) world:', (-Rc.as_matrix()[:, 2]).round(4), ' up(+Y):', Rc.as_matrix()[:, 1].round(4), ' right(+X):', Rc.as_matrix()[:, 0].round(4))
rel = Rf.inv() * Rc
print('camera in flange frame: pos', (Rf.inv().apply(cam_p - flange_p)).round(4))
print('camera rot in flange frame matrix\n', rel.as_matrix().round(4))
print('camera view dir in flange frame', (-rel.as_matrix()[:, 2]).round(4))
fc = np.array([1.3, 0, 1.6]); fq = [0.5963678059445466, 0.37992820378709397, 0.37992820378709397, 0.5963678059445466]
Rfc = q2R(fq)
print('front cam view dir', (-Rfc.as_matrix()[:, 2]).round(4), 'up', Rfc.as_matrix()[:, 1].round(4), 'right', Rfc.as_matrix()[:, 0].round(4))
