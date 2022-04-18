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
import pandas as pd
import scipy.io as sio
import pymap3d as pm
import matplotlib.pyplot as plt  # used for plotting
from mpl_toolkits.mplot3d import Axes3D
import relacaoAngQuat as AngQuat # Relação entre Ângulos e Quatérnios
import func_EKF_Abord01_v1_0 as EKF # Funções do EKF - Abordagem 01
import time
import random
import csv

# -----------------------------------------------------------------------------
''' Inicialização de Variáveis '''
# Variáveis de Tempo
tempo=0
tempoAnt = 0    # 
count = 0       # Contagem de Interações
Stop = -np.inf  # Interações para Finalização

tAmostragem = 100   # Tempo mínimo para amostragem
tCorrecao = 100      # Tempo de Correção (baseado no tAmostragem)
count = 0#tCorrecao-1

nCalibração = 100   # Amostras para Calibração

# Variáveis de Estado:
pos = np.array([[0],[0],[0]])       # Posições X, Y e Z da IMU
vel = np.array([[0],[0],[0]])       # Velocidades X, Y e Z da IMU
ang = np.array([[0],[0],[0]])       # Ângulos Yaw, Pitch, Roll
q = AngQuat.Euler2Quaternion(ang)   # Quartérnios W, X, Y e Z
bias_Gyr = np.array([[0.002438596491228], [-0.00090350877193], [0.001271929824561]])  # Bias associado ao Giroscópio
bias_Acc = np.array([[0.031291228070176], [-0.143131578947368], [-9.98327719298247]]) # Bias associado ao Aceletrômetro
bias_Acc[2] += 9.8 # Correção da Gravidade
bias_Mag = np.array([[-0.154228070175439], [-0.275666666666667], [-0.349736842105263]]) # Bias associado ao Aceletrômetro
ruido_Pos = np.array([1.76011196086875E-09, 2.04085660618074E-10, 53.909273567769])
ruido_Vel = np.array([0,0,0])
ruido_Ang = np.array([1.33892251758415E-05, 8.29680704727458E-08, 5.95242047805602E-08])
ruido_Gyr = np.array([6.90886508306164E-07, 4.06536252134762E-07, 5.48015059773327E-05])   # Ruídos do Giroscópio
ruido_Acc = np.array([0.00077661036485, 0.001583586427573, 0.000550623369042])   # Ruídos do Acelerômetro
ruido_Mag = np.array([7.64663872069556E-06, 2.31268436578172E-06, 3.89473684210522E-06])   # Ruídos do Acelerômetro
ruido_BiasGyr = 0*np.array([0.001113486842105, -0.000429824561404, -0.000969846491228]) # Ruídos do Bias do Giroscópio
ruido_BiasAcc = 0*np.array([0.273529605263158, 0.192131578947368, 0.021373026315795]) # Ruídos do Bias do Acelerômetro

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
R = np.diag(np.hstack([ruido_Pos, ruido_Vel, ruido_Ang])) #[varPos, varAng]

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
H = np.vstack([np.hstack([ Eye, Zero, Zero, Zero, Zero]),
               np.hstack([Zero,  Eye, Zero, Zero, Zero]),
               np.hstack([Zero, Zero,  Eye, Zero, Zero])])

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
''' Inicialização Leitura Arquivo .csv '''
data = pd.read_csv('22_4_14_Caminhada.csv',sep = ",");

''' Arquivo CSV '''
header = ['Tempo',
          'Pos_Lat', 'Pos_Lon', 'Pos_Alt',
          'Pos_Lon_Est', 'Pos_Lat_Est', 'Pos_Alt_Est',
          'Ang_Yaw', 'Ang_Pitch', 'Ang_Roll',
          'Ang_Yaw_Est', 'Ang_Pitch_Est', 'Ang_Roll_Est']
csv_file = open('22_4_14_Caminhada_Estimado.csv', mode='w')
writer = csv.writer(csv_file, delimiter=',')
writer.writerow(header)
csv_file.close()

pos0 = data['Pos_Lat'][0], data['Pos_Lon'][0], data['Pos_Alt'][0]
for index, row in data.iterrows():
    ''' Sinais de Entrada '''
    tempo = np.array(row['Tempo'])/5
    posPix = np.array(pm.geodetic2ned(row['Pos_Lat'], row['Pos_Lon'], row['Pos_Alt'], pos0[0], pos0[1], pos0[2])).reshape(3,1)
    angPix = np.array([[row['Ang_Yaw']], [row['Ang_Pitch']], [row['Ang_Roll']]])
    
    acc = 9.8*1e-3*np.array([[row['Xacc']], [row['Yacc']], [row['Zacc']]])
    gyro = 1e-3*np.array([[row['Xgyro']], [row['Ygyro']], [row['Zgyro']]])
    
    if tempoAnt == 0:
        tempo0 = tempo
        tempoAnt = tempo
        posAnt = np.array(posPix)
        
        ''' Pose Inicial '''
        x[0:3] = np.array(posPix)
        x[6:10] = AngQuat.Euler2Quaternion(angPix)
    
    dt = 1e-6*(tempo -tempoAnt)
    """
    Propagação do EKF
    """
    x, P, Qd = EKF.PropagaEKF(x, gyro, acc, ruidos, grav, dt, F, G, P, Q, Qd)
    
    """Correção do EKF"""
    if(count == tCorrecao):
        count=0
        
        # Variáveis de Entrada
        pos_s = np.array(posPix)
        vel_s = np.array((pos_s -posAnt))/dt
        ori_s = np.array(angPix)
        
        zSens = np.vstack([pos_s, vel_s, ori_s])
        
        x, P = EKF.CorrecaoEKF(x, zSens, P, H, R)
        posAnt = np.array(pos_s)

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
    
    histTempo.append(1e-6*(tempo-tempo0))
        
    ''' Atualização do Tempo '''
    tempoAnt = tempo
    count = count+1
    
    ''' Arquivo CSV '''
    # header = ['Tempo',
    #           'Pos_Lat', 'Pos_Lon', 'Pos_Alt',
    #           'Pos_Lon_Est', 'Pos_Lat_Est', 'Pos_Alt_Est',
    #           'Ang_Yaw', 'Ang_Pitch', 'Ang_Roll',
    #           'Ang_Yaw_Est', 'Ang_Pitch_Est', 'Ang_Roll_Est']
    csv_file = open('22_4_14_Caminhada_Estimado.csv', mode='a')
    writer = csv.writer(csv_file)
    writer.writerow([tempo,
                     posPix[0].item(), posPix[1].item(), posPix[2].item(),
                     x[0].item(), x[1].item(), x[2].item(),
                     angPix[0].item(), angPix[1].item(), angPix[2].item(),
                     angx[0], angx[1], angx[2]])
    csv_file.close()

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
ax11 = fig.add_subplot(3,2,1, title="Ângulos", ylabel="Yaw")
ax11.plot(histTempo, np.array(histRefAng)[:,0], 'r', histTempo, np.array(histAng)[:,0], 'g')
# ax21.plot(histTempo, np.array(histAng)[:,0] +3*np.array(histP)[:,0], '--m')
# ax21.plot(histTempo, np.array(histAng)[:,0] -3*np.array(histP)[:,0], '--m')
ax21 = fig.add_subplot(3,2,3, ylabel="Pitch")
ax21.plot(histTempo, np.array(histRefAng)[:,1], 'r', histTempo, np.array(histAng)[:,1], 'g')
# ax21.plot(histTempo, np.array(histAng)[:,1] +3*np.array(histP)[:,7], '--m')
# ax21.plot(histTempo, np.array(histAng)[:,1] -3*np.array(histP)[:,7], '--m')
ax31 = fig.add_subplot(3,2,5, ylabel="Roll")
ax31.plot(histTempo, np.array(histRefAng)[:,2], 'r', histTempo, np.array(histAng)[:,2], 'g')
# ax31.plot(histTempo, np.array(histAng)[:,2] +3*np.array(histP)[:,8], '--m')
# ax31.plot(histTempo, np.array(histAng)[:,2] -3*np.array(histP)[:,8], '--m')

erroAngX = np.array(histAng) -np.array(histRefAng)
for index, val in np.ndenumerate(erroAngX):
    if abs(val)>180:
        erroAngX[index] = val-np.sign(val)*360

ax12 = fig.add_subplot(3,2,2, title="Erro do Ângulos")
ax12.plot(histTempo, erroAngX[:,0], 'g')
# ax12.plot(histTempo, +3*np.array(histP)[:,6], '--m')
# ax12.plot(histTempo, -3*np.array(histP)[:,6], '--m')
ax22 = fig.add_subplot(3,2,4)
ax22.plot(histTempo, erroAngX[:,1], 'g')
# ax22.plot(histTempo, +3*np.array(histP)[:,7], '--m')
# ax22.plot(histTempo, -3*np.array(histP)[:,7], '--m')
ax32 = fig.add_subplot(3,2,6)
ax32.plot(histTempo, erroAngX[:,2], 'g')
# ax32.plot(histTempo, +3*np.array(histP)[:,8], '--m')
# ax32.plot(histTempo, -3*np.array(histP)[:,8], '--m')


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
ax11.plot(histTempo, np.array(histPos)[:,0] +3*np.array(histP)[:,0], '--m')
ax11.plot(histTempo, np.array(histPos)[:,0] -3*np.array(histP)[:,0], '--m')
ax21 = fig.add_subplot(3,2,3, ylabel="Eixo Y")
ax21.plot(histTempo, np.array(histRefPos)[:,1], 'r', histTempo, np.array(histPos)[:,1], 'g')
ax21.plot(histTempo, np.array(histPos)[:,1] +3*np.array(histP)[:,1], '--m')
ax21.plot(histTempo, np.array(histPos)[:,1] -3*np.array(histP)[:,1], '--m')
ax31 = fig.add_subplot(3,2,5, ylabel="Eixo Z")
ax31.plot(histTempo, np.array(histRefPos)[:,2], 'r', histTempo, np.array(histPos)[:,2], 'g')
ax31.plot(histTempo, np.array(histPos)[:,2] +3*np.array(histP)[:,2], '--m')
ax31.plot(histTempo, np.array(histPos)[:,2] -3*np.array(histP)[:,2], '--m')

ax12 = fig.add_subplot(3,2,2, title="Velocidade")
ax12.plot(histTempo, np.array(histVel)[:,0], 'g')
ax12.plot(histTempo, np.array(histVel)[:,0] +3*np.array(histP)[:,3], '--m')
ax12.plot(histTempo, np.array(histVel)[:,0] -3*np.array(histP)[:,3], '--m')
ax22 = fig.add_subplot(3,2,4)
ax22.plot(histTempo, np.array(histVel)[:,1], 'g')
ax22.plot(histTempo, np.array(histVel)[:,1] +3*np.array(histP)[:,4], '--m')
ax22.plot(histTempo, np.array(histVel)[:,1] -3*np.array(histP)[:,4], '--m')
ax32 = fig.add_subplot(3,2,6)
ax32.plot(histTempo, np.array(histVel)[:,2], 'g')
ax32.plot(histTempo, np.array(histVel)[:,2] +3*np.array(histP)[:,5], '--m')
ax32.plot(histTempo, np.array(histVel)[:,2] -3*np.array(histP)[:,5], '--m')

''' Plot 3D '''
fig = plt.figure()
ax = fig.add_subplot(1,2,1, title="3D", projection='3d')
ax.plot(np.array(histRefPos)[:, 0], np.array(histRefPos)[:, 1], np.array(histRefPos)[:, 2], 'r')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax = fig.add_subplot(1,2,2, title="3D", projection='3d')
ax.plot(np.array(histPos)[:, 0], np.array(histPos)[:, 1], np.array(histPos)[:, 2], 'g')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

''' Plot Planos '''
fig = plt.figure()
ax12 = fig.add_subplot(3,2,1, title="Referência", ylabel="Plano XY")
ax12.plot(np.array(histRefPos)[:, 0], np.array(histRefPos)[:, 1], 'r')
# ax12.plot(histTempo, np.array(histVel)[:,0] +3*np.array(histP)[:,6], '--m')
# ax12.plot(histTempo, np.array(histVel)[:,0] -3*np.array(histP)[:,6], '--m')
ax22 = fig.add_subplot(3,2,3, ylabel="Plano XZ")
ax22.plot(np.array(histRefPos)[:, 0], np.array(histRefPos)[:, 2], 'r')
# ax22.plot(histTempo, np.array(histVel)[:,1] +3*np.array(histP)[:,7], '--m')
# ax22.plot(histTempo, np.array(histVel)[:,1] -3*np.array(histP)[:,7], '--m')
ax32 = fig.add_subplot(3,2,5, ylabel="Plano YZ")
ax32.plot(np.array(histRefPos)[:, 1], np.array(histRefPos)[:, 2], 'r')
# ax32.plot(histTempo, np.array(histVel)[:,2] +3*np.array(histP)[:,8], '--m')
# ax32.plot(histTempo, np.array(histVel)[:,2] -3*np.array(histP)[:,8], '--m')


ax42 = fig.add_subplot(3,2,2, title="Estimado", ylabel="Plano XY")
ax42.plot(np.array(histPos)[:, 0], np.array(histPos)[:, 1], 'g')
# ax42.plot(histTempo, np.array(histVel)[:,0] +3*np.array(histP)[:,0], '--m')
# ax42.plot(histTempo, np.array(histVel)[:,0] -3*np.array(histP)[:,0], '--m')
ax52 = fig.add_subplot(3,2,4, ylabel="Plano XZ")
ax52.plot(np.array(histPos)[:, 0], np.array(histPos)[:, 2], 'g')
# ax52.plot(histTempo, np.array(histVel)[:,1] +3*np.array(histP)[:,1], '--m')
# ax52.plot(histTempo, np.array(histVel)[:,1] -3*np.array(histP)[:,1], '--m')
ax62 = fig.add_subplot(3,2,6, ylabel="Plano YZ")
ax62.plot(np.array(histPos)[:, 1], np.array(histPos)[:, 2], 'g')
# ax62.plot(histTempo, np.array(histVel)[:,2] +3*np.array(histP)[:,2], '--m')
# ax62.plot(histTempo, np.array(histVel)[:,2] -3*np.array(histP)[:,2], '--m')


''' Biases '''
fig = plt.figure(figsize=(15,10))
ax11 = fig.add_subplot(3,2,1, title="Bias Giroscópio", ylabel="Eixo X")
ax11.plot(histTempo, np.array(histBias)[:,0], 'g')
ax11.plot(histTempo, np.array(histBias)[:,0] +3*np.array(histP)[:,9], '--m')
ax11.plot(histTempo, np.array(histBias)[:,0] -3*np.array(histP)[:,9], '--m')
ax21 = fig.add_subplot(3,2,3, ylabel="Eixo Y")
ax21.plot(histTempo, np.array(histBias)[:,1], 'g')
ax21.plot(histTempo, np.array(histBias)[:,1] +3*np.array(histP)[:,10], '--m')
ax21.plot(histTempo, np.array(histBias)[:,1] -3*np.array(histP)[:,10], '--m')
ax31 = fig.add_subplot(3,2,5, ylabel="Eixo Z")
ax31.plot(histTempo, np.array(histBias)[:,2], 'g')
ax31.plot(histTempo, np.array(histBias)[:,2] +3*np.array(histP)[:,11], '--m')
ax31.plot(histTempo, np.array(histBias)[:,2] -3*np.array(histP)[:,11], '--m')

ax12 = fig.add_subplot(3,2,2, title="Bias Acelerômetro")
ax12.plot(histTempo, np.array(histBias)[:,3], 'g')
ax12.plot(histTempo, np.array(histBias)[:,3] +3*np.array(histP)[:,12], '--m')
ax12.plot(histTempo, np.array(histBias)[:,3] -3*np.array(histP)[:,12], '--m')
ax22 = fig.add_subplot(3,2,4)
ax22.plot(histTempo, np.array(histBias)[:,4], 'g')
ax22.plot(histTempo, np.array(histBias)[:,4] +3*np.array(histP)[:,13], '--m')
ax22.plot(histTempo, np.array(histBias)[:,4] -3*np.array(histP)[:,13], '--m')
ax32 = fig.add_subplot(3,2,6)
ax32.plot(histTempo, np.array(histBias)[:,5], 'g')
ax32.plot(histTempo, np.array(histBias)[:,5] +3*np.array(histP)[:,14], '--m')
ax32.plot(histTempo, np.array(histBias)[:,5] -3*np.array(histP)[:,14], '--m')