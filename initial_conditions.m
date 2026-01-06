L = 200.0;      % [mm]
T = 30.0;       % [days]
p.L = L;

Nx = 400;                  
x  = linspace(0, L, Nx);      

p.D    = 0.126;   
p.rho  = 1.0;  
p.cmax = 1.0;     

p.A     = 0.1;       
p.x0    = p.L / 2;    
p.sigma = 10;        

c0 = p.A * exp( - (x - p.x0).^2 / (2 * p.sigma^2) );

figure;
plot(x, c0, 'LineWidth', 2);
xlabel('Spatial coordinate x [mm]');
ylabel('Population density c(x,0)');
title('Initial Gaussian distribution of bacterial population');
grid on;

eqn = ['$c(x,0) = c_0 \exp\!\left(-\frac{(x-x_0)^2}{2\sigma^2}\right)$'];

text(0.55*L, 0.8*max(c0), eqn, ...
    'Interpreter','latex', ...
    'FontSize',14, ...
    'BackgroundColor','white', ...
    'EdgeColor','black');