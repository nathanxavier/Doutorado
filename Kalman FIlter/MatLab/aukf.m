function [xkk, Pxxkk, inovk, Pyykk_1] = aukf(xk_1k_1, Pk_1k_1, Qk_1, Rk, yk, nameffun, namehfun, dt);

xk_1k_1 = xk_1k_1(:);
yk = yk(:);

n = size(Pk_1k_1, 2);        

% UKF parameters:

nsp = 2*n;
w = (1/nsp)*ones(1, nsp); % Weight vectors

% FORECAST STEP:

% Sigma-point matrix:
P_root = sqrt(n).*chol(Pk_1k_1)';
Xk_1k_1 = [P_root -P_root] + repmat(xk_1k_1, 1, nsp);

% Propagating the sigma points:
Xkk_1 = feval(nameffun, Xk_1k_1);
xkk_1 = Xkk_1*w';

Pxxkk_1 = (Xkk_1 - repmat(xkk_1,1,size(Xkk_1,2))).*repmat(w,size(Xkk_1,1),1)*(Xkk_1 - repmat(xkk_1,1,size(Xkk_1,2)))' + Qk_1;

P_root = sqrt(n).*chol(Pxxkk_1)';
Xkk_1 = [P_root -P_root] + repmat(xkk_1, 1, nsp);
Ykk_1 = feval(namehfun, Xkk_1);

ykk_1 = Ykk_1*w'; 
Pxykk_1 = (Xkk_1 - repmat(xkk_1,1,size(Xkk_1,2))).*repmat(w,size(Xkk_1,1),1)*(Ykk_1 - repmat(ykk_1,1,size(Ykk_1,2)))';
Pyykk_1 = (Ykk_1 - repmat(ykk_1,1,size(Ykk_1,2))).*repmat(w,size(Ykk_1,1),1)*(Ykk_1 - repmat(ykk_1,1,size(Ykk_1,2)))' + Rk;

% DATA-ASSIMILATION STEP:

inovk = yk - ykk_1;
Kk = Pxykk_1*inv(Pyykk_1); 
xkk = xkk_1 + Kk*inovk;      % vetor de estados e parametros estimados      
Pxxkk = Pxxkk_1 - (Kk* Pyykk_1 *Kk');  % Covariancia 
