import numpy as np
import pybullet as p

def quaternion_multiply(quaternion1, quaternion0):
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array([   x1*w0 + y1*z0 - z1*y0 + w1*x0,
                        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                        x1*y0 - y1*x0 + z1*w0 + w1*z0,
                        -x1*x0 - y1*y0 - z1*z0 + w1*w0])

def quaternion_inverse(quaternion):
    
    return np.hstack([-quaternion[:3],quaternion[-1]])

def passive_rotation(quaternion1,quaternion2):
    '''
    Orientation only
    O2/W = O1/W * O2/O1
    '''
    qua = quaternion_multiply(quaternion_multiply(quaternion1,quaternion2),quaternion_inverse(quaternion1))
    return qua/np.linalg.norm(qua)

def active_rotation(quaternion1,quaternion2):
    '''
    Orientation only
    O2/O1 = O1/W * O2/W
    '''
    qua = quaternion_multiply(quaternion_multiply(quaternion_inverse(quaternion1),quaternion2),quaternion1)
    return qua/np.linalg.norm(qua)

def bullet_passive_rotation(pos_qua1,pos_qua2):
    '''
    Orientation only
    O2/W = O1/W * O2/O1
    '''
    return p.multiplyTransforms(*p.multiplyTransforms(*pos_qua1,*pos_qua2),*p.invertTransform(*pos_qua1))

def bullet_active_rotation(pos_qua1,pos_qua2):
    '''
    Orientation only
    O2/O1 = O1/W * O2/W
    '''
    return p.multiplyTransforms(*p.multiplyTransforms(*p.invertTransform(*pos_qua1),*pos_qua2),*pos_qua1)

def bullet_passive_cor_mul(pos_qua1,pos_qua2):
    '''
    Cordinate change
    O2/W = O1/W * O2/O1
    '''
    return p.multiplyTransforms(*pos_qua1,*pos_qua2)

def bullet_active_cor_mul(pos_qua1,pos_qua2):
    '''
    Cordinate change
    O2/O1 = O1/W * O2/W
    '''
    return p.multiplyTransforms(*p.invertTransform(*pos_qua1),*pos_qua2)

# # # TEST CODE # # #
# qua = np.array([1,0,0,1])
# vec = np.array([0,0,1,0])
# result = passive_rotation(qua,vec)
# inp = active_rotation(qua,result)
# print(result)
# print(inp)