% Simula��o de um rob� diferencial com erros de atua��o. O Rob� � equipado
% com um sensor global tipo GPS

clear all
close all

N=3000;   % Numero de interacoes
t=0;      % Tempo
global dt; dt=0.01;  % Periodo de amostragem do robo
cont1=0;  % Controla o tempo do controlador e amostrador de dados
cont2=0;  % Controla o tempo do GPS  

j=1;  % contador para o Filtro e controlador
k=1;  % contador para o GPS

global u; u=[0; 0];         % Sinal de controle
xr=[0.01; -0.02; 0.1]; % Configura��o inicial do rob� 
y=[0; 0];         % Sinal inicial do GPS  

% Inicializa��o do Unscented Filtro de Kalman
xk=[0; 0; 0]; %
P = diag([0.05 0.05 0.05]);
Pdiag(:,1) = diag(P);
cons1y(:,1) = 2.71;
cons1x(:,1) = 4.61;
C = [1 0 0 ; 0 1 0];
S = zeros(2,1);
Q = [.02 0 0 ; 0 .01 0 ; 0 0 .001];
R = [.3 0 ; 0 .5];
Rk = diag(R);
tol_cons = 1e-16;
perc = 0.95;

tic
for i=1:N
    cont1=cont1+1;
    
    if cont1>=10  % Controlador e filtro - a cada 10 intera��es do rob� ocorre uma amostragem do controlador e do filtro
        cont1=0;
        cont2=cont2+1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % GPS - medi��o
        if cont2>=10 % A cada 10 intera��es do controlador e do filtro o GPS fornece um dado
            cont2=0;
                       
            y=xr(1:2)+0.02*y+randn([2 1])*0.25;
            
            % Log do GPS
            Ty(k)=t;
            Y(k,:)=y;
            
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
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Unscented Filtro de Kalman 
        
        [xk, P, S, Rk] = aukf(xk',P,Q, R, y, 'eq_robo_n_holonomico', 'eq_obs_n_holonomico', dt);
        Pdiag = [Pdiag diag(P)];

        % Log do Filtro            
        Tk(j)=t;
        Xk(j,:)=xk;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        j=j+1;
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Dinamica do rob�

    u(1)=u(1)*1.01 + 0.05*randn([1 1])*u(1); % erros de atua��o na velocidade linear: erro no raio da rodas + erro aleat�rio (derrapagem, etc)
    u(2)=u(2)*0.98 - u(1)*0.015 + 0.02*randn([1 1]); % erros de atua��o na velocidade angular: atrito que impede girar + diferen�a de velocidade das rodas + erro aleat�rio
   
   if abs(u(1))>0.85            % Satura��o de velocidade linear m�xima - V
        u(1)=0.85*sign(u(1));
    end
    if abs(u(2))>0.4           % Satura��o de velocidade angular m�xima - w
        u(2)=0.4*sign(u(2));
    end
    xr(1)=xr(1)+dt*u(1)*cos(xr(3));
    xr(2)=xr(2)+dt*u(1)*sin(xr(3));
    xr(3)=xr(3)+dt*u(2);
    
    % Log do rob�
    Tr(i)=t;
    Xr(i,:)=xr;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Desenha em tempo real - comentar para ficar mais r�pido
%     figure(1)
%     plot(xr(1), xr(2), 'o');
%     hold on
%     plot(Xr(:,1),Xr(:,2) )
%     xlabel('x(m)')
%     ylabel('y(m)')
%     axis([-2 10 -2 10])
%     %drawnow
%     hold off
    
    t=t+dt;
end
toc

% Desenha o resultado final
figure(1)
plot(xr(1), xr(2),'o');
xlabel('x(m)')
ylabel('y(m)')
axis([-2 10 -2 10])
hold on
robo=plot(Xr(:,1),Xr(:,2),'r' );
gps=plot(Y(:,1),Y(:,2), 'x');
plot(Y(:,1),Y(:,2), '--')
filtro=plot(Xk(:,1),Xk(:,2), '.-');
legend([robo, gps, filtro], 'Caminho Real', 'GPS', 'Predi��o','Location','NorthWest')