"""Planar kinematics / state estimation for Hopper-v5 built only from public model.xml facts.
Absolute segment angles (rotation about +y, positive = top tilts toward +x):
  a1 = pitch - q_thigh, a2 = a1 - q_leg, a3 = a2 - q_foot.
A 'down' vector of length L in a segment with angle a is (-L sin a, -L cos a).
Foot capsule axis (d,0,0) -> (d cos a3, -d sin a3); heel at d=-0.13, toe at d=+0.26 from ankle, radius 0.06.
"""
import numpy as np

MASS = np.array([3.66519, 4.05789, 2.78136, 5.31557])  # torso, thigh, leg, foot (density 1000 capsules)
MTOT = float(MASS.sum())
G = 9.81


def kin(z, p, q1, q2, q3, x=0.0):
    a1 = p - q1
    a2 = a1 - q2
    a3 = a2 - q3
    s, c = np.sin, np.cos
    hip = np.array([x - 0.2 * s(p), z - 0.2 * c(p)])
    d1 = np.array([-s(a1), -c(a1)])
    d2 = np.array([-s(a2), -c(a2)])
    f = np.array([c(a3), -s(a3)])
    knee = hip + 0.45 * d1
    ankle = knee + 0.5 * d2
    heel = ankle - 0.13 * f
    toe = ankle + 0.26 * f
    com = (MASS[0] * np.array([x, z]) + MASS[1] * (hip + 0.225 * d1)
           + MASS[2] * (knee + 0.25 * d2) + MASS[3] * (ankle + 0.065 * f)) / MTOT
    return dict(hip=hip, knee=knee, ankle=ankle, heel=heel, toe=toe, com=com, a=(a1, a2, a3))


def estimate(obs):
    o = np.asarray(obs, dtype=float)
    z, p, q1, q2, q3, vx, vz, dp, dq1, dq2, dq3 = o[:11]
    K0 = kin(z, p, q1, q2, q3)
    eps = 1e-5
    K1 = kin(z + eps * vz, p + eps * dp, q1 + eps * dq1, q2 + eps * dq2, q3 + eps * dq3, x=eps * vx)
    st = dict(z=z, p=p, q1=q1, q2=q2, q3=q3, vx=vx, vz=vz, dp=dp, dq1=dq1, dq2=dq2, dq3=dq3)
    for k in ('hip', 'knee', 'ankle', 'heel', 'toe', 'com'):
        st[k] = K0[k]
        st[k + '_v'] = (K1[k] - K0[k]) / eps
    st['a1'], st['a2'], st['a3'] = K0['a']
    st['da1'] = dp - dq1
    st['da2'] = st['da1'] - dq2
    st['da3'] = st['da2'] - dq3
    st['heel_h'] = K0['heel'][1] - 0.06
    st['toe_h'] = K0['toe'][1] - 0.06
    st['ankle_h'] = K0['ankle'][1] - 0.04
    st['foot_h'] = min(st['heel_h'], st['toe_h'], st['ankle_h'])
    # COM relative to ankle (horizontal) and COM height
    st['xc'] = st['com'][0] - st['ankle'][0]
    st['zc'] = st['com'][1]
    st['vxc'] = st['com_v'][0]
    st['vzc'] = st['com_v'][1]
    return st
