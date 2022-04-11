% Simulação de um robô holonômico com erros de atuação. O Robô é equipado
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
xr=[0.01; -0.02]; % Configuração inicial do robô 
y=[0; 0];         % Sinal inicial do GPS  

A = [1 0; 0 1];
B = [1 0 ; 0 1];
C = [1 0; 0 1];
z = [0;0];
P = [1 0 ; 0 1];
S = [0 0 ; 0 0];
K = [0 ; 0];
Q = [.35 0 ; 0 .5];
R = [.15 0 ; 0 .25];

% Inicialização do Filtro de Kalman
xk=[0; 0]; %
x=xk;

figure(1)

for i=1:N
    cont1=cont1+1;
    
    if cont1>=10  % Controlador e filtro - a cada 10 interações do robô ocorre uma amostragem do controlador e do filtro
        cont1=0;
        cont2=cont2+1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % GPS - medição
        if cont2>=10 % A cada 10 interações do controlador e do filtro o GPS fornece um dado
                       
            y=xr+0.02*y+randn([2 1])*0.25;
            
            % Log do GPS
            Ty(k)=t;
            Y(k,:)=y;
            
            k=k+1;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
               
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Controlador
        if t<10
            u=[0; 0.7];
        elseif t<20
            u=[0.6; 0];
        else
            u=[0; -0.6];
        end
                
        % Log do Controlador
        Tu(j)=t;
        U(j,:)=u;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Filtro de Kalman
        % Predição
        xk = A*x + B*u*dt;
        P = A*P*A' + Q;
        
        if cont2>=10
            cont2=0;
            %Atualização
            yk = y -C*xk;
            S = C*P*C' + R;
            K = P*C'*inv(S);
            
            xk = xk+K*yk;
            P = (eye(2)-K*C)*P;
            
            % Log Inovação
            In(k-1,:)=[yk' 3*P(1,1) 3*P(2,2)];
        end
        x = xk;
        
        desenha_elipse(P, xk);
        
        % Log do Filtro            
        Tk(j)=t;
        Xk(j,:)=xk;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
        
        j=j+1;
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Dinamica do robô

    u=u*1.01 +[0 0.02;0.03 0]*u +0.05*randn([2 2])*u; % erros de atuação: erro no raio da rodas + acoplamento entre vx e vy + erro aleatório (derrapagem, etc)
    if abs(u(1))>0.85            % Saturação de velocidade máxima - Vx
        u(1)=0.85*sign(u(1));
    end
    if abs(u(2))>0.85            % Saturação de velocidade máxima - Vy
        u(2)=0.85*sign(u(2));
    end
    xr=xr+dt*u;
    
    % Log do robô
    Tr(i)=t;
    Xr(i,:)=xr;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Desenha em tempo real - comentar para ficar mais rápido
    hold on
    plot(xr(1), xr(2), 'ob');
    plot(Xr(:,1),Xr(:,2), '-r' )
    xlabel('x(m)')
    ylabel('y(m)')
    axis([-2 10 -2 10])
    drawnow
    hold off
    
    t=t+dt;
end

% Desenha o resultado final
figure(1), 
plot(xr(1), xr(2),'ob');
xlabel('x(m)')
ylabel('y(m)')
axis([-2 10 -2 10])
hold on
robo=plot(Xr(:,1),Xr(:,2),'r' );
gps=plot(Y(:,1),Y(:,2), 'x');
plot(Y(:,1),Y(:,2), '--')
filtro=plot(Xk(:,1),Xk(:,2), '.-');
legend([robo, gps, filtro], 'Caminho Real', 'GPS', 'Predição','Location','NorthWest')

figure, subplot(211), plot(Ty,In(:,1),'b', Ty,In(:,3),'r', Ty,-In(:,3),'r')
        subplot(212), plot(Ty,In(:,2),'b', Ty,In(:,4),'r', Ty,-In(:,4),'r')