function desenha_elipse(P, xc)

th=linspace(0, 2*pi, 50);
y=[cos(th); sin(th)];
x=(sqrtm(P)*y)';
x(:,1)=x(:,1)+xc(1);
x(:,2)=x(:,2)+xc(2);

plot(x(:,1), x(:,2),'g');

xlabel('x')
ylabel('y')