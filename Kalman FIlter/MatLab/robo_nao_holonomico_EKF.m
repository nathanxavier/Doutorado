% Simulação de um robô diferencial com erros de atuação. O Robô é equipado
% com um sensor global tipo GPS

clear all
close all

N=3000;   % Numero de interacoes
t=0;      % Tempo
dt=0.01;  % Periodo de amostragem do robo

cont1=0;  % Controla o tempo do controlador e amostrador de dados
cont2=0;  % Controla o tempo do GPS  

j=1;  % contador para o Filtro e controlador
k=1;  % contador para o GPS

u=[0; 0];         % Sinal de controle
xr=[0.01; -0.02; 0.1]; % Configuração inicial do robô 
y=[0; 0];         % Sinal inicial do GPS  

% Inicialização do Filtro de Kalman
xk=[0; 0; 0]; %
x=xk;
f = [xk(1)+dt*10*u(1)*cos(xk(3)) ; xk(2)+dt*10*u(1)*sin(xk(3)) ; xk(3)+dt*10*u(2)];
F = [1 0 1 ; 0 1 xk(3) ; 0 0 1];
h = [xk(1) ; xk(2)];
H = [1 0 0 ; 0 1 0];
P = ones(3);
S = zeros(2);
K = zeros(3,2);
Q = [.05 0 0 ; 0 .01 0 ; 0 0 .02];
R = [.08 0; 0 .09];

flag = 1;

figure(1), hold on

tic
for i=1:N
    cont1=cont1+1;
    
    if cont1>=10  % Controlador e filtro - a cada 10 interações do robô ocorre uma amostragem do controlador e do filtro
        cont1=0;
        cont2=cont2+1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % GPS - medição
        if cont2>=10 % A cada 10 interações do controlador e do filtro o GPS fornece um dado
%             cont2=0;
                       
            y=xr(1:2)+0.02*y+randn([2 1])*0.0001;
            
            % Log do GPS
            Ty(k)=t;
            Y(k,:)=y;
            
               plot(y)

            k=k+1;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
               
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Controlador
        if t<5
            u=[0.7; 0.0];
        elseif t<20
            u=[0.5; 0.2];
        else
            u=[0.4; -0.1];
        end
        
        % Log do Controlador
        Tu(j)=t;
        U(j,:)=u;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Filtro de Kalman vem aqui
        
        % Predição
        xk= [xk(1)+dt*10*u(1)*cos(xk(3)); xk(2)+dt*10*u(1)*sin(xk(3)); xk(3)+dt*10*u(2)];
        F = [1 0 -sin(xk(3)) ; 0 1 cos(xk(3)) ; 0 0 1];
        P = F*P*F' + Q;
        
        if cont2>=10
            cont2=0;
            %Atualização
            yk = y -[xk(1) ; xk(2)];
            S = H*P*H' + R;
            K = P*H'*inv(S);
            
            xk = xk+K*yk;
            P = (eye(3)-K*H)*P;
            
            % Log Inovação
            In(k-1,:)=[yk' 3*P(1,1) 3*P(2,2)];
            
            
        end
        x = xk;
        
        desenha_elipse([P(1,1) P(1,2) ; P(2,1) P(2,2)], [xk(1) xk(2)]);
        
        % Log do Filtro            
        Tk(j)=t;
        Xk(j,:)=xk;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
        
        j=j+1;
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Dinamica do robô

    u(1)=u(1)*1.01 + 0.05*randn([1 1])*u(1); % erros de atuação na velocidade linear: erro no raio da rodas + erro aleatório (derrapagem, etc)
    u(2)=u(2)*0.98 - u(1)*0.015 + 0.02*randn([1 1]); % erros de atuação na velocidade angular: atrito que impede girar + diferença de velocidade das rodas + erro aleatório
   
   if abs(u(1))>0.85            % Saturação de velocidade linear máxima - V
        u(1)=0.85*sign(u(1));
    end
    if abs(u(2))>0.4           % Saturação de velocidade angular máxima - w
        u(2)=0.4*sign(u(2));
    end
    xr(1)=xr(1)+dt*u(1)*cos(xr(3));
    xr(2)=xr(2)+dt*u(1)*sin(xr(3));
    xr(3)=xr(3)+dt*u(2);
    
    
    
%     if t > 5 && flag == 1
%         xr = xr + [10 ; 0 ; .7];
%         flag = 2;
%     elseif t > 10 && flag == 2
%         xr = xr + [0 ; 10 ; -.9];
%         flag = 3;
%     elseif t > 15 && flag == 3
%         xr = xr + [-10 ; -10 ; -.9];
%         flag = 0;
%     end
    % Log do robô
    Tr(i)=t;
    Xr(i,:)=xr;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Desenha em tempo real - comentar para ficar mais rápido
%     hold on
%     plot(xr(1), xr(2), 'ob');
%     plot(Xr(:,1),Xr(:,2), 'r' )
%     xlabel('x(m)')
%     ylabel('y(m)')
%     axis ([-2 20 -10 20])
%     drawnow
%     hold off
    
    t=t+dt;
end
toc

figure
plot(xr(1), xr(2),'o');
xlabel('x(m)')
ylabel('y(m)')
axis tight
hold on
robo=plot(Xr(:,1),Xr(:,2),'r' );
gps=plot(Y(:,1),Y(:,2), 'x');
plot(Y(:,1),Y(:,2), '--')
filtro=plot(Xk(:,1),Xk(:,2), '.-');
legend([robo, gps, filtro], 'Caminho Real', 'GPS', 'Predição','Location','NorthWest')

% Desenha o resultado final
figure, hold on
plot(xr(1), xr(2),'o');
xlabel('x(m)')
ylabel('y(m)')
axis tight
hold on
robo=plot(Xr(:,1),Xr(:,2),'r' );
gps=plot(Y(:,1),Y(:,2), 'x');
plot(Y(:,1),Y(:,2), '--')
filtro=plot(Xk(:,1),Xk(:,2), '.-');
legend([robo, gps, filtro], 'Caminho Real', 'GPS', 'Predição','Location','NorthWest')

figure, subplot(211), plot(Ty,In(:,1),'b', Ty,In(:,3),'r', Ty,-In(:,3),'r')
        subplot(212), plot(Ty,In(:,2),'b', Ty,In(:,4),'r', Ty,-In(:,4),'r')

% figure, hold on, plot(Tk, Xk(:,1)), plot(Tr, Xr(:,1)), plot(Ty, Y(:,1)), legend('Robô','Filtro','Medição')
% figure, hold on, plot(Tk, Xk(:,2)), plot(Tr, Xr(:,2)), plot(Ty, Y(:,2)), legend('Robô','Filtro','Medição')
