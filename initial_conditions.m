%% Domain definition
L = 200.0;      % [mm]
T = 30.0;       % [days]
p.L = L;

Nx = 400;                      % Number of spatial grid points
x  = linspace(0, L, Nx);       % Spatial domain [0, L]

%% Parameters of the equation
p.D    = 0.126;   % Diffusion coefficient [mm^2/day]
p.rho  = 1.0;     % Growth rate [day^-1]
p.cmax = 1.0;     % Carrying capacity

%% Parameters of the initial Gaussian
p.A     = 0.1;        % Initial amplitude
p.x0    = p.L / 2;    % Center of the Gaussian
p.sigma = 10;         % Width of the Gaussian

%% Initial condition
c0 = p.A * exp( - (x - p.x0).^2 / (2 * p.sigma^2) );

%% Plot
figure;
plot(x, c0, 'LineWidth', 2);
xlabel('Spatial coordinate x [mm]');
ylabel('Population density c(x,0)');
title('Initial Gaussian distribution of bacterial population');
grid on;

% Equation as LaTeX text
eqn = ['$c(x,0) = c_0 \exp\!\left(-\frac{(x-x_0)^2}{2\sigma^2}\right)$'];

text(0.55*L, 0.8*max(c0), eqn, ...
    'Interpreter','latex', ...
    'FontSize',14, ...
    'BackgroundColor','white', ...
    'EdgeColor','black');