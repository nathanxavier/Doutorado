#------------------------------------------------------------------------------
# Created By  : Nathan - Piter-N
# Created Date: 2022/Mar
# version ='1.0'
# -----------------------------------------------------------------------------
"""
Implementação e testes da Abordagem 01
"""
# -----------------------------------------------------------------------------
"""
Referência:
    Heterogeneous multi-sensor fusion for 2d and 3d pose estimation
        Hanieh Deilamsalehy, 2017
    Métodos e Técnicas de Fusão de Dados para Navegação Aérea.
        Relatório Tècnico, Experimento 7
    A Kalman Filter-Based Algorithm for IMU-Camera Calibration: Observability Analysis and Performance Evaluation
        Faraz Mirzaei & Stergios Roumeliotis, 2008
"""
# -----------------------------------------------------------------------------
# Bibliotecas
# import pandas as pd
import numpy as np  # used for general matrix computation
import scipy.io as sio
import matplotlib.pyplot as plt  # used for plotting
from mpl_toolkits.mplot3d import Axes3D
from dronekit import connect # Pixhawk
import comuPixhawk # Coleta de dados da Pixhawk
import relacaoAngQuat as AngQuat # Relação entre Ângulos e Quatérnios
import func_EKF_Abord01_v1_0 as EKF # Funções do EKF - Abordagem 01
import time
import random

# -----------------------------------------------------------------------------
''' Inicialização de Variáveis '''
# Variáveis de Tempo
tempo=0
tempoAnt = 0    # 
count = 0       # Contagem de Interações
Stop = -np.inf  # Interações para Finalização

tAmostragem = 100   # Tempo mínimo para amostragem
tCorrecao = np.inf      # Tempo de Correção (baseado no tAmostragem)

nCalibração = 100   # Amostras para Calibração

# Variáveis de Estado:
pos = np.array([[-45.8844543],[-23.2025882],[763.1]])    # Posições X, Y e Z da IMU
vel = np.array([[0],[0],[0]])      # Velocidades X, Y e Z da IMU
ang = np.array([[0],[0],[0]])     # Ângulos Yaw, Pitch, Roll
q = np.array([[1],[0],[0],[0]])    # Quartérnios W, X, Y e Z
bias_Gyr = 0*np.array([[1],[1],[1]])  # Bias associado ao Giroscópio
bias_Acc = 0*np.array([[1],[1],[1]]) # Bias associado ao Aceletrômetro
# bias_Acc[2] += 0*9.8 # Correção da Gravidade
ruido_Pos = 0*np.array([1,1,1])
ruido_Ang = 0*np.array([1,1,1])
ruido_Gyr = 0*np.array([1,1,1])   # Ruídos do Giroscópio
ruido_Acc = 0*np.array([1,1,1])   # Ruídos do Acelerômetro
ruido_BiasGyr = 0*np.array([1,1,1]) # Ruídos do Bias do Giroscópio
ruido_BiasAcc = 0*np.array([1,1,1]) # Ruídos do Bias do Acelerômetro

# Variáveis de Sensores
acc = np.array([[0],[0],[0]])   # Medições do Acelerômetro
gyro = np.array([[0],[0],[0]])  # Medições do Giroscópio
mag = np.array([[0],[0],[0]])   # Medições do Magnetômetro
gps = np.array([[0],[0],[0]])   # Medições do GPS
grav = np.array([[0],[0],[9.8]]) # Vetor de Gravidade

# Vetores de Estado
x = np.vstack([pos, vel, q, bias_Gyr, bias_Acc])           # Estados do Sistema
xcorr = x

xtilde = np.vstack([pos, vel, ang, bias_Gyr, bias_Acc]) # Erro dos Estados

#Vetor de Ruídos
ruidos = np.hstack([ruido_Gyr, ruido_BiasGyr, ruido_Acc, ruido_BiasAcc]).reshape([12,1]) # Estados do Sistema

zSens = np.zeros(6)

# Matriz de Covariância dos Estados
P = xtilde*xtilde.T

# Matriz de Covariância dos Ruídos do Modelo (Contínuo)
Q = np.diag(np.hstack([ruido_Gyr, ruido_BiasGyr, ruido_Acc, ruido_BiasAcc]))

# Matriz de Covariância dos Ruídos do Modelo (Discreto)
Qd = np.eye(15)

# Matriz de Covariância do Erro de Observação
R = np.diag(np.hstack([ruido_Pos]))#, ruido_Ang])) #[varPos, varAng]

''' Matrizes do Filtro de Kalman - Abordagem 01 '''
# Matriz de Transição
Eye = np.eye(3)
Eye15 = np.eye(15)
Zero = np.zeros([3,3])

''' Mirzaei & Roumeliotis, 2008 '''
Rq = AngQuat.MatrixQuaternion(x[6:10])
Sacc = EKF.Skew(acc-x[13:16])
RqSacc = Rq*Sacc

Sgyro = EKF.Skew(gyro-x[10:13])

F = np.vstack([np.hstack([Zero,  Eye, Zero, Zero, Zero]),
               np.hstack([Zero, Zero, -RqSacc, Zero, -Rq]),
               np.hstack([Zero, Zero, -Sgyro, Zero, -Eye]),
               np.hstack([Zero, Zero, Zero, Zero, Zero]),
               np.hstack([Zero, Zero, Zero, Zero, Zero])])
               

G = np.vstack([np.hstack([Zero,  Eye, Zero, Zero]),
               np.hstack([Zero, Zero, -Rq, Zero]),
               np.hstack([-Eye, Zero, Zero, Zero]),
               np.hstack([Zero, Eye,  Zero, Zero]),
               np.hstack([Zero, Zero, Zero,  Eye])])

# Matriz de Observação
H = np.vstack([np.hstack([ Eye, Zero, Zero, Zero, Zero])])
               # np.hstack([Zero, Zero,  Eye, Zero, Zero])])

''' Logs das Variáveis '''
# Tempo
histTempo = []

# Sensores
histAcc = []
histGyro = []
histZ = []

# Ângulos
histRefAng = []
histAng = []

# Quatérnios
histQuat = []
histRefQuat = []

# Posição
histRefPos = []
histPos = []

# Velocidade
histVel = []

# Bias
histBias = []

# Gravidade
histGrav = []

# Matriz P
histP = []

# -----------------------------------------------------------------------------
''' Dados MatLab '''
data = sio.loadmat('check.mat')
AspB = data['accBody']
wB_bi = data['angVelBody']
q = np.array([data['qA'][0, 0], data['qB'][0, 0], data['qC'][0, 0], data['qD'][0, 0]]).reshape([4,1])
dt = data['dt'].item()
pos = data['initPos']
initVel = data['initVel']

pos = data['initPos'].reshape([3,1])
vel = data['initVel'].reshape([3,1])
ang = np.array([[0],[0],[0]])     # Ângulos Yaw, Pitch, Roll

x = np.vstack([pos, vel, q, bias_Gyr, bias_Acc])           # Estados do Sistema

for i in range(len(AspB)):
    ''' Sinais de Entrada '''
    # tempo = Valores.time_usec
    posPix = np.array([[0],[0],[0]])
    angPix = np.array([[0],[0],[0]])
    randAcc = 0*np.array([random.random()-.5, random.random()-.5, random.random()-.5])
    randGyr = 0*np.array([random.random()-.5, random.random()-.5, random.random()-.5])
    acc = AspB[i] +randAcc
    acc = acc.reshape([3,1])
    gyro = wB_bi[i] +randGyr
    gyro = gyro.reshape([3,1])
    
    # dt = 1e-6*(tempo -tempoAnt)
    
    """
    Propagação do EKF
    """
    x, P, Qd = EKF.PropagaEKF(x, gyro, acc, ruidos, grav, dt, F, G, P, Q, Qd)
    
# =============================================================================
#     """Correção do EKF"""
#     if(count == tCorrecao):
#         count=0
#         
#         # Variáveis de Entrada
#         pos_s = np.array([[posPix.lat], [posPix.lon], [posPix.alt]])
#         vel_s = np.array((pos_s -posAnt))/dt
#         ori_s = np.array([[angPix.yaw], [angPix.pitch], [angPix.roll]])
#         
#         zSens = np.vstack([pos_s, vel_s, ori_s])
#         
#         x, P = EKF.CorrecaoEKF(x, zSens, P, H, R)
#         posAnt = np.array(pos_s)
# =============================================================================

    ''' Logs das Variáveis '''
    # Sensores
    histAcc.append(np.ravel(acc))
    histGyro.append(np.ravel(gyro))
    histZ.append(np.ravel(zSens))
        
    # Posição
    histRefPos.append(np.ravel(posPix))
    histPos.append(np.ravel(x[0:3]))
    
    # Velocidade
    histVel.append(np.ravel(x[3:6]))
    
    # Ângulos
    angx = AngQuat.Quaternion2Euler(x[6:10])
    histRefAng.append(np.rad2deg(np.ravel(angPix)))
    histAng.append(np.rad2deg(np.ravel(angx)))
    
    # Quatérnios
    qRef = AngQuat.Euler2Quaternion(np.array([angPix]))
    histRefQuat.append(np.ravel(qRef))
    histQuat.append(np.ravel(x[6:10]))
    
    # Bias
    histBias.append(np.ravel(x[10:16]))
    
    # # Gravidade
    # histGrav.append(np.ravel(x[15:18]))
    
    diag = []
    for i in range(len(P)):
        diag.append(abs(P[i,i]))
    histP.append(np.array(np.sqrt(diag)))
    
    histTempo.append(1e-6*(tempo))
        
    ''' Atualização do Tempo '''
    tempo += dt
    count = count+1

''' Tempo '''
fig = plt.figure(figsize=(15,10))
plt.plot(histTempo, 'g')


''' Aceleração e Giro '''
fig = plt.figure(figsize=(15,10))
ax11 = fig.add_subplot(3,2,1, title="Aceleração", ylabel="Eixo X")
ax11.plot(histTempo, np.array(histAcc)[:,0], 'g')
ax21 = fig.add_subplot(3,2,3, ylabel="Eixo Y")
ax21.plot(histTempo, np.array(histAcc)[:,1], 'g')
ax31 = fig.add_subplot(3,2,5, ylabel="Eixo Z")
ax31.plot(histTempo, np.array(histAcc)[:,2], 'g')

ax12 = fig.add_subplot(3,2,2, title="Giroscópio")
ax12.plot(histTempo, np.array(histGyro)[:,0], 'g')
ax22 = fig.add_subplot(3,2,4)
ax22.plot(histTempo, np.array(histGyro)[:,1], 'g')
ax32 = fig.add_subplot(3,2,6)
ax32.plot(histTempo, np.array(histGyro)[:,2], 'g')


''' Ângulos '''
fig = plt.figure(figsize=(15,10))
ax11 = fig.add_subplot(3,2,1, title="Ângulos", ylabel="Roll")
ax11.plot(histTempo, np.array(histRefAng)[:,0], 'r', histTempo, np.array(histAng)[:,0], 'g')
# ax11.plot(histTempo, np.array(histAng)[:,0] +3*np.array(histP)[:,0], '--m')
# ax11.plot(histTempo, np.array(histAng)[:,0] -3*np.array(histP)[:,0], '--m')
ax21 = fig.add_subplot(3,2,3, ylabel="Pitch")
ax21.plot(histTempo, np.array(histRefAng)[:,1], 'r', histTempo, np.array(histAng)[:,1], 'g')
# ax21.plot(histTempo, np.array(histAng)[:,1] +3*np.array(histP)[:,1], '--m')
# ax21.plot(histTempo, np.array(histAng)[:,1] -3*np.array(histP)[:,1], '--m')
ax31 = fig.add_subplot(3,2,5, ylabel="Yaw")
ax31.plot(histTempo, np.array(histRefAng)[:,2], 'r', histTempo, np.array(histAng)[:,2], 'g')
# ax31.plot(histTempo, np.array(histAng)[:,2] +3*np.array(histP)[:,2], '--m')
# ax31.plot(histTempo, np.array(histAng)[:,2] -3*np.array(histP)[:,2], '--m')

erroAngX = np.array(histAng) -np.array(histRefAng)
for index, val in np.ndenumerate(erroAngX):
    if abs(val)>180:
        erroAngX[index] = val-np.sign(val)*360

ax12 = fig.add_subplot(3,2,2, title="Erro do Ângulos")
ax12.plot(histTempo, erroAngX[:,0], 'g')
ax22 = fig.add_subplot(3,2,4)
ax22.plot(histTempo, erroAngX[:,1], 'g')
ax32 = fig.add_subplot(3,2,6)
ax32.plot(histTempo, erroAngX[:,2], 'g')


''' Quatérnios '''
fig = plt.figure(figsize=(15,10))
ax11 = fig.add_subplot(2,2,1, title="q[0]")
ax11.plot(histTempo, np.array(histRefQuat)[:,0], 'r', histTempo, np.array(histQuat)[:,0], 'g')
ax21 = fig.add_subplot(2,2,2, title="q[1]")
ax21.plot(histTempo, np.array(histRefQuat)[:,1], 'r', histTempo, np.array(histQuat)[:,1], 'g')
ax31 = fig.add_subplot(2,2,3, title="q[2]")
ax31.plot(histTempo, np.array(histRefQuat)[:,2], 'r', histTempo, np.array(histQuat)[:,2], 'g')
ax41 = fig.add_subplot(2,2,4, title="q[3]")
ax41.plot(histTempo, np.array(histRefQuat)[:,3], 'r', histTempo, np.array(histQuat)[:,3], 'g')


''' Posição e Velocidade '''
fig = plt.figure(figsize=(15,10))
ax11 = fig.add_subplot(3,2,1, title="Posição", ylabel="Eixo X")
ax11.plot(histTempo, np.array(histRefPos)[:,0], 'r', histTempo, np.array(histPos)[:,0], 'g')
# ax11.plot(histTempo, np.array(histPos)[:,0] +3*np.array(histP)[:,3], '--m')
# ax11.plot(histTempo, np.array(histPos)[:,0] -3*np.array(histP)[:,3], '--m')
ax21 = fig.add_subplot(3,2,3, ylabel="Eixo Y")
ax21.plot(histTempo, np.array(histRefPos)[:,1], 'r', histTempo, np.array(histPos)[:,1], 'g')
# ax21.plot(histTempo, np.array(histPos)[:,1] +3*np.array(histP)[:,4], '--m')
# ax21.plot(histTempo, np.array(histPos)[:,1] -3*np.array(histP)[:,4], '--m')
ax31 = fig.add_subplot(3,2,5, ylabel="Eixo Z")
ax31.plot(histTempo, np.array(histRefPos)[:,2], 'r', histTempo, np.array(histPos)[:,2], 'g')
# ax31.plot(histTempo, np.array(histPos)[:,2] +3*np.array(histP)[:,5], '--m')
# ax31.plot(histTempo, np.array(histPos)[:,2] -3*np.array(histP)[:,5], '--m')

ax12 = fig.add_subplot(3,2,2, title="Velocidade")
ax12.plot(histTempo, np.array(histVel)[:,0], 'g')
# ax12.plot(histTempo, np.array(histVel)[:,0] +3*np.array(histP)[:,6], '--m')
# ax12.plot(histTempo, np.array(histVel)[:,0] -3*np.array(histP)[:,6], '--m')
ax22 = fig.add_subplot(3,2,4)
ax22.plot(histTempo, np.array(histVel)[:,1], 'g')
# ax22.plot(histTempo, np.array(histVel)[:,1] +3*np.array(histP)[:,7], '--m')
# ax22.plot(histTempo, np.array(histVel)[:,1] -3*np.array(histP)[:,7], '--m')
ax32 = fig.add_subplot(3,2,6)
ax32.plot(histTempo, np.array(histVel)[:,2], 'g')
# ax32.plot(histTempo, np.array(histVel)[:,2] +3*np.array(histP)[:,8], '--m')
# ax32.plot(histTempo, np.array(histVel)[:,2] -3*np.array(histP)[:,8], '--m')

fig = plt.figure(figsize=(15,10))
ax11 = fig.add_subplot(2,2,1, title="3D")
graph3D_ned = Axes3D(ax11, auto_add_to_figure=False)
graph3D_ned.set_xlabel('North (m)')
graph3D_ned.set_ylabel('East (m)')
graph3D_ned.set_zlabel('Down (m)')
ax11.plot(np.array(histPos)[:, 0], np.array(histPos)[:, 1], np.array(histPos)[:, 2], color='green', markersize=2)

ax12 = fig.add_subplot(3,2,2, title="Velocidade", ylabel="Plano XY")
ax12.plot(np.array(histPos)[:, 0], np.array(histPos)[:, 1], 'g')
# ax12.plot(histTempo, np.array(histVel)[:,0] +3*np.array(histP)[:,6], '--m')
# ax12.plot(histTempo, np.array(histVel)[:,0] -3*np.array(histP)[:,6], '--m')
ax22 = fig.add_subplot(3,2,4, ylabel="Plano XZ")
ax22.plot(np.array(histPos)[:, 0], np.array(histPos)[:, 2], 'g')
# ax22.plot(histTempo, np.array(histVel)[:,1] +3*np.array(histP)[:,7], '--m')
# ax22.plot(histTempo, np.array(histVel)[:,1] -3*np.array(histP)[:,7], '--m')
ax32 = fig.add_subplot(3,2,6, ylabel="Plano YZ")
ax32.plot(np.array(histPos)[:, 1], np.array(histPos)[:, 2], 'g')
# ax32.plot(histTempo, np.array(histVel)[:,2] +3*np.array(histP)[:,8], '--m')
# ax32.plot(histTempo, np.array(histVel)[:,2] -3*np.array(histP)[:,8], '--m')


''' Biases '''
fig = plt.figure(figsize=(15,10))
ax11 = fig.add_subplot(3,2,1, title="Bias Giroscópio", ylabel="Eixo X")
ax11.plot(histTempo, np.array(histBias)[:,0], 'g')
# ax11.plot(histTempo, np.array(histBias)[:,0] +3*np.array(histP)[:,9], '--m')
# ax11.plot(histTempo, np.array(histBias)[:,0] -3*np.array(histP)[:,9], '--m')
ax21 = fig.add_subplot(3,2,3, ylabel="Eixo Y")
ax21.plot(histTempo, np.array(histBias)[:,1], 'g')
# ax21.plot(histTempo, np.array(histBias)[:,1] +3*np.array(histP)[:,10], '--m')
# ax21.plot(histTempo, np.array(histBias)[:,1] -3*np.array(histP)[:,10], '--m')
ax31 = fig.add_subplot(3,2,5, ylabel="Eixo Z")
ax31.plot(histTempo, np.array(histBias)[:,2], 'g')
# ax31.plot(histTempo, np.array(histBias)[:,2] +3*np.array(histP)[:,11], '--m')
# ax31.plot(histTempo, np.array(histBias)[:,2] -3*np.array(histP)[:,11], '--m')

ax12 = fig.add_subplot(3,2,2, title="Bias Acelerômetro")
ax12.plot(histTempo, np.array(histBias)[:,3], 'g')
# ax12.plot(histTempo, np.array(histBias)[:,3] +3*np.array(histP)[:,12], '--m')
# ax12.plot(histTempo, np.array(histBias)[:,3] -3*np.array(histP)[:,12], '--m')
ax22 = fig.add_subplot(3,2,4)
ax22.plot(histTempo, np.array(histBias)[:,4], 'g')
# ax22.plot(histTempo, np.array(histBias)[:,4] +3*np.array(histP)[:,13], '--m')
# ax22.plot(histTempo, np.array(histBias)[:,4] -3*np.array(histP)[:,13], '--m')
ax32 = fig.add_subplot(3,2,6)
ax32.plot(histTempo, np.array(histBias)[:,5], 'g')
# ax32.plot(histTempo, np.array(histBias)[:,5] +3*np.array(histP)[:,14], '--m')
# ax32.plot(histTempo, np.array(histBias)[:,5] -3*np.array(histP)[:,14], '--m')