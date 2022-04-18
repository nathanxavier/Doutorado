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
import matplotlib.pyplot as plt  # used for plotting
from dronekit import connect # Pixhawk
import comuPixhawk # Coleta de dados da Pixhawk
import relacaoAngQuat as AngQuat # Relação entre Ângulos e Quatérnios
import func_EKF_Abord01_v1_0 as EKF # Funções do EKF - Abordagem 01
import time
import random

# -----------------------------------------------------------------------------
''' Inicialização de Variáveis '''
# Variáveis de Tempo
tempo = 0
dt=0
tempoAnt = 0        #
Stop = 100          # Interações para Finalização

tAmostragem = 5e-2  # Tempo mínimo para amostragem
tCorrecao = np.inf#20      # Tempo de Correção
count = 0#tCorrecao-1 # Contagem de Interações
countStop = 0

nCalibração = 10   # Amostras para Calibração

# Variáveis de Estado:
pos = np.array([[-45.8844543],[-23.2025882],[763.1]])    # Posições X, Y e Z da IMU
vel = np.array([[0],[0],[0]])      # Velocidades X, Y e Z da IMU
ang = np.array([[0],[0],[0]])     # Ângulos Yaw, Pitch, Roll
q = np.array([[1],[0],[0],[0]])    # Quartérnios W, X, Y e Z
bias_Gyr = np.array([[ 0.00064],[-0.00036],[0.00011]])  # Bias associado ao Giroscópio
bias_Acc = np.array([[-0.109270],[-0.110838],[-9.928184]]) # Bias associado ao Aceletrômetro
bias_Acc[2] += 9.8 # Correção da Gravidade
ruido_Pos = np.array([1e-4,1e-4,1e-4])
ruido_Vel = np.array([0,0,0])
ruido_Ang = np.array([5.43209046e-08, 3.76306402e-08, 3.04091209e-08])
ruido_Gyr = np.array([4.504e-07, 2.704e-07, 9.790e-08])   # Ruídos do Giroscópio
ruido_Acc = np.array([1.81275500e-04, 7.62461560e-05, 1.54970144e-04])   # Ruídos do Acelerômetro
ruido_BiasGyr = np.array([1e-4,1e-4,1e-4]) # Ruídos do Bias do Giroscópio
ruido_BiasAcc = np.array([1e-2,1e-2,1e-2]) # Ruídos do Bias do Acelerômetro

# Variáveis de Sensores
acc = np.array([[0],[0],[0]])   # Medições do Acelerômetro
gyro = np.array([[0],[0],[0]])  # Medições do Giroscópio
mag = np.array([[0],[0],[0]])   # Medições do Magnetômetro
gps = np.array([[0],[0],[0]])   # Medições do GPS
grav = np.array([[0],[0],[9.8]]) # Vetor de Gravidade

# Vetores de Estado
x = np.vstack([pos, vel, q, bias_Gyr, bias_Acc])        # Estados do Sistema
xcorr = np.array(x)

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
R = np.diag(np.hstack([ruido_Pos, ruido_Vel, ruido_Ang]))

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
               np.hstack([Zero, Eye, Zero, Zero, Zero]),
               np.hstack([Zero, Zero,  Eye, Zero, Zero])])

''' Logs das Variáveis '''
histTempo = []

histAcc = []
histGyro = []
histRefPos = []

histX = []
histXprop = []
histXcorr = []

histZ = []

histAngPix = []
histQuatPix = []
histAngXprop = []
histAngXcorr = []

histP = []

# -----------------------------------------------------------------------------
''' Inicialização da Comunicação com a Pixhawk'''
connection_string = '/dev/ttyACM0' # Linux

# Connect to the Vehicle
print('Connecting to vehicle on: %s' % connection_string)
vehicle = connect(connection_string, wait_ready=True, vehicle_class=comuPixhawk.MyVehicle)
# Criação do Callback do IMU
Valores = comuPixhawk.raw_imu_callback(vehicle, 'raw_imu', vehicle.raw_imu)
print('Conectado!')

''' Pose Inicial '''
print('Calibração...')

Calibracao = []
amostra = 0
while(len(Calibracao)<nCalibração):
    while(amostra == Valores.time_usec):
        pass
    calPos = np.array([vehicle.location.global_frame.lat, vehicle.location.global_frame.lon, vehicle.location.global_frame.alt])
    calAng = np.array([vehicle.attitude.yaw, vehicle.attitude.pitch, vehicle.attitude.roll])
    calGyr = 1e-3*np.array([Valores.xgyro, Valores.ygyro, Valores.zgyro])
    calAcc = 9.8*1e-3*np.array([Valores.xacc, Valores.yacc, Valores.zacc])
    Calibracao.append([calPos, calAng, calGyr, calAcc])
    amostra = Valores.time_usec

if(nCalibração>0):
    # Variáveis de Estado:
    pos = np.array(Calibracao)[:,0].mean(0).reshape([3,1])
    ang = np.array(Calibracao)[:,1].mean(0).reshape([3,1])
    bias_Gyr = np.array(Calibracao)[:,2].mean(0).reshape([3,1])
    bias_Acc = np.array(Calibracao)[:,3].mean(0).reshape([3,1])
    bias_Acc[2] += 9.8 # Correção da Gravidade
    ruido_Pos = np.array(Calibracao)[:,0].var(0)
    ruido_Ang = np.array(Calibracao)[:,1].var(0)
    ruido_Gyr = np.array(Calibracao)[:,2].var(0)
    ruido_Acc = np.array(Calibracao)[:,3].var(0)
    
    # Vetores de Estado
    q = AngQuat.Euler2Quaternion(ang)
    x = np.vstack([pos, vel, q, bias_Gyr, bias_Acc])           # Estados do Sistema
    xtilde = np.vstack([pos, vel, ang, bias_Gyr, bias_Acc]) # Erro dos Estados
    ruidos = np.hstack([ruido_Gyr, ruido_BiasGyr, ruido_Acc, ruido_BiasAcc]).reshape([12,1]) # Estados do Sistema
    
    P = xtilde*xtilde.T
    Q = np.diag(np.hstack([ruido_Gyr, ruido_BiasGyr, ruido_Acc, ruido_BiasAcc]))
    R = np.diag(np.hstack([ruido_Pos, ruido_Vel, ruido_Ang]))
    print('Calibrado!')

print('Início dos Testes')
while True:
    ''' Sinais de Entrada '''
    # Sistema aguarda nova medição do IMU para propagação
    while(tempoAnt == Valores.time_usec):
        pass
    tempo = Valores.time_usec
    posPix = vehicle.location.global_frame
    angPix = vehicle.attitude
    acc = 9.8*1e-3*np.array([[Valores.xacc], [Valores.yacc], [Valores.zacc]]) # Medições do Acelerômetro
    gyro = 1e-3*np.array([[Valores.xgyro], [Valores.ygyro], [Valores.zgyro]]) # Medições do Giroscópio
    
    if tempoAnt == 0:
        tempo0 = tempo
        tempoAnt = tempo
        posAnt = np.array([[posPix.lat], [posPix.lon], [posPix.alt]])
        
    dt = 1e-6*(tempo -tempoAnt)

    """
    Propagação do EKF
    """
    x, P, Qd = EKF.PropagaEKF(x, gyro, acc, ruidos, grav, dt, F, G, P, Q, Qd)
    
    """Correção do EKF"""
    if(count == tCorrecao):
        count=0
        
        # Variáveis de Entrada
        pos_s = np.array([[posPix.lat], [posPix.lon], [posPix.alt]])
        vel_s = np.array((pos_s -posAnt))/dt
        ori_s = np.array([[angPix.yaw], [angPix.pitch], [angPix.roll]])
        
        zSens = np.vstack([pos_s, vel_s, ori_s])
        
        x, P = EKF.CorrecaoEKF(x, zSens, P, H, R)
        posAnt = np.array(pos_s)

    ''' Logs das Variáveis '''
    histAcc.append(np.ravel(acc))
    histGyro.append(np.ravel(gyro))
    histRefPos.append(np.array([[posPix.lat], [posPix.lon], [posPix.alt]]))
    
    histXprop.append(np.ravel(x))
    histXcorr.append(np.ravel(xcorr))
    histZ.append(np.ravel(zSens))
    
    angx = AngQuat.Quaternion2Euler(x[6:10])
    histAngXprop.append(np.rad2deg(np.ravel(angx)))
    
    angxcorr = AngQuat.Quaternion2Euler(xcorr[6:10])
    histAngXcorr.append(np.rad2deg(np.ravel(angxcorr)))
    
    quat = AngQuat.Euler2Quaternion(np.array([angPix.yaw, angPix.pitch, angPix.roll]))
    histQuatPix.append(np.ravel(quat))
    histAngPix.append(np.rad2deg(np.array([angPix.yaw, angPix.pitch, angPix.roll])))
    
    diag = []
    for i in range(len(P)):
        diag.append(abs(P[i,i]))
    histP.append(np.array(np.sqrt(diag)))
    
    histTempo.append(1e-6*(tempo-tempo0))
    
    ''' Atualização do Tempo '''
    tempoAnt = tempo
    count = count+1
    countStop = countStop +1
    # time.sleep(1/tAmostragem)
    if(countStop >= Stop):
        break

vehicle.close()

''' Tempo '''
fig = plt.figure(figsize=(15,10))
plt.plot(histTempo)


''' Aceleração e Giro '''
fig = plt.figure(figsize=(15,10))
ax11 = fig.add_subplot(3,2,1, title="Aceleração", ylabel="Eixo X")
ax11.plot(histTempo, np.array(histAcc)[:,0])
ax21 = fig.add_subplot(3,2,3, ylabel="Eixo Y")
ax21.plot(histTempo, np.array(histAcc)[:,1])
ax31 = fig.add_subplot(3,2,5, ylabel="Eixo Z")
ax31.plot(histTempo, np.array(histAcc)[:,2])

ax12 = fig.add_subplot(3,2,2, title="Giroscópio")
ax12.plot(histTempo, np.array(histGyro)[:,0])
ax22 = fig.add_subplot(3,2,4)
ax22.plot(histTempo, np.array(histGyro)[:,1])
ax32 = fig.add_subplot(3,2,6)
ax32.plot(histTempo, np.array(histGyro)[:,2])


''' Posição e Velocidade '''
fig = plt.figure(figsize=(15,10))
ax11 = fig.add_subplot(3,2,1, title="Posição", ylabel="Latitude")
ax11.plot(histTempo, np.array(histRefPos)[:,0], 'r', histTempo, np.array(histXprop)[:,0], 'g')
# ax11.plot(histTempo, np.array(histXprop)[:,0] +3*np.array(histP)[:,0], '--m')
# ax11.plot(histTempo, np.array(histXprop)[:,0] -3*np.array(histP)[:,0], '--m')
ax21 = fig.add_subplot(3,2,3, ylabel="Longitude")
ax21.plot(histTempo, np.array(histRefPos)[:,1], 'r', histTempo, np.array(histXprop)[:,1], 'g')
# ax21.plot(histTempo, np.array(histXprop)[:,1] +3*np.array(histP)[:,1], '--m')
# ax21.plot(histTempo, np.array(histXprop)[:,1] -3*np.array(histP)[:,1], '--m')
ax31 = fig.add_subplot(3,2,5, ylabel="Altitude")
ax31.plot(histTempo, np.array(histRefPos)[:,2], 'r', histTempo, np.array(histXprop)[:,2], 'g')
# ax31.plot(histTempo, np.array(histXprop)[:,2] +3*np.array(histP)[:,2], '--m')
# ax31.plot(histTempo, np.array(histXprop)[:,2] -3*np.array(histP)[:,2], '--m')

ax12 = fig.add_subplot(3,2,2, title="Velocidade")
ax12.plot(histTempo, np.array(histXprop)[:,3], 'g')
# ax12.plot(histTempo, np.array(histXprop)[:,3] +3*np.array(histP)[:,3], '--m')
# ax12.plot(histTempo, np.array(histXprop)[:,3] -3*np.array(histP)[:,3], '--m')
ax22 = fig.add_subplot(3,2,4)
ax22.plot(histTempo, np.array(histXprop)[:,4], 'g')
# ax22.plot(histTempo, np.array(histXprop)[:,4] +3*np.array(histP)[:,4], '--m')
# ax22.plot(histTempo, np.array(histXprop)[:,4] -3*np.array(histP)[:,4], '--m')
ax32 = fig.add_subplot(3,2,6)
ax32.plot(histTempo, np.array(histXprop)[:,5], 'g')
# ax32.plot(histTempo, np.array(histXprop)[:,5] +3*np.array(histP)[:,5], '--m')
# ax32.plot(histTempo, np.array(histXprop)[:,5] -3*np.array(histP)[:,5], '--m')


''' Ângulos '''
fig = plt.figure(figsize=(15,10))
ax11 = fig.add_subplot(3,2,1, title="Ângulos", ylabel="Yaw")
ax11.plot(histTempo, np.array(histAngPix)[:,0], 'r', histTempo, np.array(histAngXprop)[:,0])
# ax11.plot(histTempo, np.array(histAngXprop)[:,0] +3*np.array(histP)[:,6], '--m')
# ax11.plot(histTempo, np.array(histAngXprop)[:,0] -3*np.array(histP)[:,6], '--m')
ax21 = fig.add_subplot(3,2,3, ylabel="Pitch")
ax21.plot(histTempo, np.array(histAngPix)[:,1], 'r', histTempo, np.array(histAngXprop)[:,1])
# ax21.plot(histTempo, np.array(histAngXprop)[:,1] +3*np.array(histP)[:,7], '--m')
# ax21.plot(histTempo, np.array(histAngXprop)[:,1] -3*np.array(histP)[:,7], '--m')
ax31 = fig.add_subplot(3,2,5, ylabel="Roll")
ax31.plot(histTempo, np.array(histAngPix)[:,2], 'r', histTempo, np.array(histAngXprop)[:,2])
# ax31.plot(histTempo, np.array(histAngXprop)[:,2] +3*np.array(histP)[:,8], '--m')
# ax31.plot(histTempo, np.array(histAngXprop)[:,2] -3*np.array(histP)[:,8], '--m')

erroAngX = np.array(histAngXprop) -np.array(histAngPix)
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
ax11.plot(histTempo, np.array(histQuatPix)[:,0], 'r', histTempo, np.array(histXprop)[:,6], 'g')
ax21 = fig.add_subplot(2,2,2, title="q[1]")
ax21.plot(histTempo, np.array(histQuatPix)[:,1], 'r', histTempo, np.array(histXprop)[:,7], 'g')
ax31 = fig.add_subplot(2,2,3, title="q[2]")
ax31.plot(histTempo, np.array(histQuatPix)[:,2], 'r', histTempo, np.array(histXprop)[:,8], 'g')
ax41 = fig.add_subplot(2,2,4, title="q[3]")
ax41.plot(histTempo, np.array(histQuatPix)[:,3], 'r', histTempo, np.array(histXprop)[:,9], 'g')


''' Biases '''
fig = plt.figure(figsize=(15,10))
ax11 = fig.add_subplot(3,2,1, title="Bias Giroscópio", ylabel="Eixo X")
ax11.plot(histTempo, np.array(histXprop)[:,10], 'g')
# ax11.plot(histTempo, np.array(histXprop)[:,10] +3*np.array(histP)[:,9], '--m')
# ax11.plot(histTempo, np.array(histXprop)[:,10] -3*np.array(histP)[:,9], '--m')
ax21 = fig.add_subplot(3,2,3, ylabel="Eixo Y")
ax21.plot(histTempo, np.array(histXprop)[:,11], 'g')
# ax21.plot(histTempo, np.array(histXprop)[:,11] +3*np.array(histP)[:,10], '--m')
# ax21.plot(histTempo, np.array(histXprop)[:,11] -3*np.array(histP)[:,10], '--m')
ax31 = fig.add_subplot(3,2,5, ylabel="Eixo Z")
ax31.plot(histTempo, np.array(histXprop)[:,12], 'g')
# ax31.plot(histTempo, np.array(histXprop)[:,12] +3*np.array(histP)[:,11], '--m')
# ax31.plot(histTempo, np.array(histXprop)[:,12] -3*np.array(histP)[:,11], '--m')

ax12 = fig.add_subplot(3,2,2, title="Bias Acelerômetro")
ax12.plot(histTempo, np.array(histXprop)[:,13], 'g')
# ax12.plot(histTempo, np.array(histXprop)[:,13] +3*np.array(histP)[:,12], '--m')
# ax12.plot(histTempo, np.array(histXprop)[:,13] -3*np.array(histP)[:,12], '--m')
ax22 = fig.add_subplot(3,2,4)
ax22.plot(histTempo, np.array(histXprop)[:,14], 'g')
# ax22.plot(histTempo, np.array(histXprop)[:,14] +3*np.array(histP)[:,13], '--m')
# ax22.plot(histTempo, np.array(histXprop)[:,14] -3*np.array(histP)[:,13], '--m')
ax32 = fig.add_subplot(3,2,6)
ax32.plot(histTempo, np.array(histXprop)[:,15], 'g')
# ax32.plot(histTempo, np.array(histXprop)[:,15] +3*np.array(histP)[:,14], '--m')
# ax32.plot(histTempo, np.array(histXprop)[:,15] -3*np.array(histP)[:,14], '--m')
