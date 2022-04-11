#------------------------------------------------------------------------------
# Created By  : Nathan - Piter-N
# Created Date: 2022/Mar
# version ='1.0'
# -----------------------------------------------------------------------------
"""
Cálculos e transformações de ângulos e quatérnios utilizando a sequência Z-Y-X (3-2-1)

- MatrixEuler:
    Retorna a Matriz de Rotação dos Ângulos
- Euler2Quaternion:
    Cálculo dos Quatérnios a partir dos Ângulos
- MatrixQuaternion:
    Retorna a Matriz de Rotação dos Quatérnios
- Quaternion2Euler:
    Cálculo dos Ângulos a partir dos Quatérnios
- QuatMultiply:
    Multiplicação de quatérnios
- SmallAngle:
    Transformação de Pequenos Ângulos para Quatérnios
- QuatInverse:
    Cálculo do Inverso do Quatérnio
"""
# -----------------------------------------------------------------------------
"""
Referência:
    Euler Angles, Quaternions and Transformation Matrices:
        https://ntrs.nasa.gov/citations/19770024290
    Representing attitude: Euler angles, unit quaternions, and rotation vectors.
        James Diebel. 2006.
"""
# -----------------------------------------------------------------------------
import numpy as np

def MatrixEuler (ang):
    ang = np.ravel(ang)
    
    # Matriz de Rotação Z-Y-X
    cy = np.cos(ang[0])
    sy = np.sin(ang[0])
    cp = np.cos(ang[1])
    sp = np.sin(ang[1])
    cr = np.cos(ang[2])
    sr = np.sin(ang[2])
    
    # Euler Angles, Quaternions and Transformation Matrices
    # M = np.array([[cy*cp, cy*sp*sr -sy*cr, cy*sp*cr +sy*sr],
    #               [sy*cp, sy*sp*sr +cy*cr, sy*sp*cr -cy*sr],
    #               [-sp, cp*sr, cp*cr]])
    
    # Quarternions and Rotation Sequences
    # Eq 4.4 (THe Aerospace Sequence)
    M = np.array([[cy*cp, sy*cp, -sp],
                  [cy*sp*sr -sy*cr, sy*sp*sr +cy*cr, cp*sr],
                  [cy*sp*cr +sy*sr, sy*sp*cr -cy*sr, cp*cr]])
    return M

def Euler2Quaternion (ang):
    ang = np.ravel(ang)
    
    # Quaternio Transformação Z-Y-X
    cy = np.cos(.5*ang[0])
    sy = np.sin(.5*ang[0])
    cp = np.cos(.5*ang[1])
    sp = np.sin(.5*ang[1])
    cr = np.cos(.5*ang[2])
    sr = np.sin(.5*ang[2])
    
    # Euler Angles, Quaternions and Transformation Matrices
    q = np.array([[cy*cp*cr +sy*sp*sr],
                  [cy*cp*sr -sy*sp*cr],
                  [cy*sp*cr +sy*cp*sr],
                  [sy*cp*cr -cy*sp*sr]])
    q = q/np.linalg.norm(q)
    return q

def MatrixQuaternion (q):
    # Matriz de Transformação entre frames por Quatérnios
    # Rotação Z-Y-X
    q = np.ravel(q)
    M = np.array([[q[0]**2 +q[1]**2 -q[2]**2 -q[3]**2, 2*(q[1]*q[2] -q[0]*q[3]), 2*(q[1]*q[3] +q[0]*q[2])],
                  [2*(q[1]*q[2] +q[0]*q[3]), q[0]**2 -q[1]**2 +q[2]**2 -q[3]**2, 2*(q[2]*q[3] -q[0]*q[1])],
                  [2*(q[1]*q[3] -q[0]*q[2]), 2*(q[2]*q[3] +q[0]*q[1]), q[0]**2 -q[1]**2 -q[2]**2 +q[3]**2]])
    
    # Quaternion to Direction Cosines
    # M = np.array([[-1 +2*(q[0]**2 +q[1]**2), 2*(q[1]*q[2] +q[0]*q[3]), 2*(q[1]*q[3] -q[0]*q[2])],
    #               [2*(q[1]*q[2] -q[0]*q[3]), -1 +2*(q[0]**2 +q[2]**2), 2*(q[2]*q[3] +q[0]*q[1])],
    #               [2*(q[1]*q[3] +q[0]*q[2]), 2*(q[2]*q[3] -q[0]*q[1]), -1 +2*(q[0]**2 +q[3]**2)]])
    return M

def Quaternion2Euler (q):
    # Euler Angles, Quaternions and Transformation Matrices
    # yaw =   np.arctan2((2*(q[1]*q[2] +q[0]*q[3])), (q[0]**2 +q[1]**2 -q[2]**2 -q[3]**2))        # atan2(m21/m11)
    # pitch = np.arctan2(-(2*(q[1]*q[3] -q[0]*q[2])), np.sqrt(1 -(2*(q[1]*q[3] -q[0]*q[2]))**2)) # atan2(-m31/sqrt(1-m31²)
    # roll =  np.arctan2((2*(q[2]*q[3] +q[0]*q[1])), (q[0]**2 -q[1]**2 -q[2]**2 +q[3]**2))      # atan2(m32/m33)
    
    # # Wikipedia
    # yaw =   np.arctan2((2*(q[1]*q[2] +q[0]*q[3])), 1 -2*(q[2]**2 +q[3]**2))
    # pitch = np.arcsin(2*(q[1]*q[3] -q[0]*q[2]))
    # roll =  np.arctan2(-(2*(q[2]*q[3] +q[0]*q[1])), 1 -2*(q[1]**2 +q[2]**2))
    
    # Quarternions and Rotation Sequences
    # 7.8 Quaternion to Euler Angles
    yaw =   np.arctan2(2*(q[1]*q[2] +q[0]*q[3]), 2*(q[0]**2 +q[1]**2) -1)
    pitch = np.arcsin(-2*(q[1]*q[3] -q[0]*q[2]))
    roll =  np.arctan2(2*(q[2]*q[3] +q[0]*q[1]), 2*(q[0]**2 +q[3]**2) -1)
    return yaw.item(), pitch.item(), roll.item()

def QuatMultiply (q0, q1):
    q0 = np.ravel(q0)
    q1 = np.ravel(q1)
    return np.array([[-q1[1]*q0[1] -q1[2]*q0[2] -q1[3]*q0[3] +q1[0]*q0[0]],
                     [ q1[1]*q0[0] +q1[2]*q0[3] -q1[3]*q0[2] +q1[0]*q0[1]],
                     [-q1[1]*q0[3] +q1[2]*q0[0] +q1[3]*q0[1] +q1[0]*q0[2]],
                     [ q1[1]*q0[2] -q1[2]*q0[1] +q1[3]*q0[0] +q1[0]*q0[3]]])

def SmallAngle (ang):
    ang = np.ravel(ang)
    return np.array([[0],
                     [.5*ang[0]],
                     [.5*ang[1]],
                     [.5*ang[2]]])

def QuatInverse (q):
    q = np.ravel(q)
    return np.array([[q[0]],
                     [-q[1]],
                     [-q[2]],
                     [-q[3]]])