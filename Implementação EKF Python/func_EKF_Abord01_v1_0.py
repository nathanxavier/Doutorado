#------------------------------------------------------------------------------
# Created By  : Nathan - Piter-N
# Created Date: 2022/Mar
# version ='1.0'
# -----------------------------------------------------------------------------
"""
Coleta de dados da Pixhawk
- Skew:
    Matriz Antissimétrica

- NominalState:
    Modelo Discreto que ignora sinais de baixa accplitude, erros e ruídos 
    como Gaussianos de Média Zero (na = nw = 0) (p. 68) (Eq. 3.8a~d)

- ErrorState:
    Prioridade de estimação e coleção dos erros, baseado na linearização da 
    dinâmica do sistema.
"""
# -----------------------------------------------------------------------------
"""
Referência:
    Heterogeneous multi-sensor fusion for 2d and 3d pose estimation
        Hanieh Deilaccsalehy, 2017
    Métodos e Técnicas de Fusão de Dados para Navegação Aérea:
        Relatório Tècnico, Experimento 7
    A Kalman Filter-Based Algorithm for IMU-Caccera Calibration: Observability Analysis and Performance Evaluation
        Faraz Mirzaei & Stergios Roumeliotis, 2008
    3D Rotation Converter
        https://www.andre-gaschler.com/rotationconverter/
"""
# -----------------------------------------------------------------------------
import numpy as np
import relacaoAngQuat as AngQuat

def Skew(var):
    var = np.ravel(var)
    return np.matrix([[0, -var[2], var[1]],
                     [var[2], 0, -var[0]],
                     [-var[1], var[0], 0]])

"""
NominalState(v, q, bw, ba, gyro, acc, g):
    Entradas:
        - Vetor de Estados (x):
            v: Velocidades X, Y e Z
            q: Quatérnio
            bw: Bias do Giroscópio
            ba: Bias do Acelerômetro
        - Medições da IMU:
            gyro: Velocidade Angular X, Y e Z do IMU
            acc: Aceleração X, Y e Z do IMU
        - g: Vetor de Gravidade
    Saída:
        - dp: Variação das Posições
        - dv: Variação das Velocidades
        - dq: Variação do Quatérnio
        - dbw: Variação do Bias do Giroscópio
        - dba: Variação do Bias do Acelerômetro
"""
def NominalState(v, q, bw, ba, gyro, acc, g):
    Rq = AngQuat.MatrixQuaternion(q)
    wbar =  np.vstack([0, (gyro-bw)])

    dp = np.array(v)
    dv = Rq*np.matrix(acc -ba) +g
    dq = .5*AngQuat.QuatMultiply(wbar, q)
    dbw = np.array([[0],[0],[0]])
    dba = np.array([[0],[0],[0]])
    
    return np.vstack([dp, dv, dq, dbw, dba])

"""
ErrorState(q, nw, nbw, na, nba):
    Entradas:
        - Vetor de Estados (x):
            q: Quatérnio
        - Vetor de Ruídos:
            nw: Ruídos na Velocidade Angular X, Y e Z
            na: Ruídos na Aceleração X, Y e Z
            nbw: Ruídos do Bias do Giroscópio
            nba: Ruídos do Bias do Acelerômetro
    Saída:
        - Vetor de Estimação (xtilde):
            pp: Variação das Posições
            pv: Variação das Velocidades
            pq: Variação do Quatérnio
            pbw: Variação do Bias do Giroscópio
            pba: Variação do Bias do Acelerômetro
"""
def ErrorState(q, nw, nbw, na, nba):
    Rq = np.matrix(AngQuat.MatrixQuaternion(q))
    
    dp = np.array(nbw)
    dv = -Rq*na
    dtheta = -np.array(nw)
    dbw = np.array(nbw)
    dba = np.array(nba)
    
    dq = AngQuat.SmallAngle(dtheta)
    return np.vstack([dp, dv, dq.reshape([4,1]), dbw, dba])

"""
PropagaEKF(x, xtilde, gyro, acc, n, g, dt, F, G, P, Q, Qd):
    Entradas:
        - Estados do Sistema (x):
            pos: Posição X, Y e Z
            vel: Velocidades X, Y e Z
            q: Quatérnio
            bw: Bias do Giroscópio
            ba: Bias do Acelerômetro
        - Medições da IMU:
            gyro: Velocidade Angular X, Y e Z do IMU
            acc: Aceleração X, Y e Z do IMU
        - Vetor de Ruídos (n):
            nw: Ruídos na Velocidade Angular X, Y e Z do IMU
            na: Ruídos na Aceleração X, Y e Z do IMU
            nbw: Ruídos do Bias do Giroscópio
            nba: Ruídos do Bias do Acelerômetro
        - Vetor de Gravidade (g)
        - Variação do Tempo (dt)
        - Matriz de Transição dos Estados (F)
        - Matriz de Transição dos Ruídos (G)
        - Matriz de Covariância dos Estados (P)
        - Matriz de Covariância dos Ruídos do Modelo (Contínuo) (Q):
            sigma_nw: Variância dos Ruídos na Velocidade Angular X, Y e Z do IMU
            sigma_na: Variância dos Ruídos na Aceleração X, Y e Z do IMU
            sigma_nbw: Variância dos Ruídos do Bias do Giroscópio
            sigma_nba: Variância dos Ruídos do Bias do Acelerômetro
        - Matriz de Covariância dos Ruídos do Modelo (Discreto) (Qd)

    Saída:
        - Estados do Sistema (x)
        - Matriz de Covariância dos Estados (P)
        - Matriz de Covariância dos Ruídos do Modelo (Discreto) (Qd)
"""
def PropagaEKF(x, gyro, acc, n, g, dt, F, G, P, Q, Qd):
    Sgyro = Skew(gyro-x[10:13])
    Rq = AngQuat.MatrixQuaternion(x[6:10])
    Sacc = Skew(acc-x[13:16])
    RqSacc = Rq*Sacc
    
    ''' Matriz de Transição '''
    F[3:6, 6:9] = -np.array(RqSacc) # F(2,3)
    F[3:6, 12:15] = -np.array(Rq)   # F(2,5)
    F[6:9, 6:9] = -np.array(Sgyro)  # F(3,3)
    
    G[3:6, 6:9] = -np.array(Rq)     # G(2,3)
    
    ''' Propagação dos Estados '''
    fx = NominalState(x[3:6], x[6:10], x[10:13], x[13:16], gyro, acc, g)
    gn = ErrorState(x[6:10], n[0:3], n[3:6], n[6:9], n[9:12])
    
    x = x +fx*dt +gn*dt
    x[6:10] = x[6:10]/np.linalg.norm(x[6:10]) # Normalização dos Quatérnios
    
    ''' Atualização da Matriz de Covariância Discreta '''
    Qd = F*G*Q*G.T*F.T
    
    ''' Propagação da Matriz de Covariância dos Estados '''
    dP = F*P*F.T +Qd
    P = P +dP*dt
    
    return x, P, Qd

"""
CorrecaoEKF(x, pos_s, ori_s, P, H, R)
    Entradas:
        - Estados do Sistema (x):

    Saída:
        - Estados do Sistema (x)
        - Erro dos Estados (xtilde)
        - Matriz de Covariância dos Estados (P)
"""
def CorrecaoEKF(x, z, P, H, R):
    ''' Matriz de Transição '''
    Eye = np.eye(15)
    
    ''' Ganho de Kalman '''
    S = H*P*H.T +R
    invS = np.linalg.inv(S)
    K = P*H.T *invS
    
    ''' Correção do Erro '''
    hx_pos = np.array(x[0:6])
    yaw, pitch, roll = AngQuat.Quaternion2Euler(x[6:10])
    hx_ang = np.array([yaw, pitch, roll]).reshape([3,1])
    hx_bias = np.array(x[9:15])
    
    hx = np.vstack([hx_pos, hx_ang, hx_bias])
    dx = K*(z -np.matrix(H)*hx)
        
    qdx = AngQuat.Euler2Quaternion(dx[6:9])
    
    x[0:6] += np.array(dx[0:6])
    x[6:10] = AngQuat.QuatMultiply(qdx, x[6:10])
    x[10:16] += np.array(dx[9:15])
    
    P = (Eye -K*H)*P
    
    return x, P