%% ==========================================================
% Heun method (explicit trapezoidal / RK2) vs Reference solutions
%
% Reference strategy (as in the report):
%  - Spatial reference: pdepe  -> compare c(x,T)
%  - Temporal reference: MOL + ode45 -> compare c(x0,t)
%  - Check pdepe vs MOL+ode45 overlap (error ~ 0)
%
% PDE: c_t = D c_xx + rho c (1 - c/cmax)
% BC:  Neumann (no-flux)
% IC:  Gaussian inoculum centered at x=L/2
%% ==========================================================

clear; close all; clc;

%% ------------------- Parameters ---------------------------
L = 200.0;      % [mm]
T = 30.0;      % [days]
p.L = L;

p.D    = 0.126; % [mm^2/days^-1]
p.rho  = 1.0; % [0.012 days^-1}
p.cmax = 1.0;
p.A = 0.1;
p.x0 = p.L / 2;
p.sigma = 10;


Nx = 800;
x  = linspace(0, L, Nx);
dx = x(2) - x(1);

% Reference time grid (as in your figures)
Nt = 20000;
t  = linspace(0, T, Nt);
dt = t(2) - t(1); % temporal step (h in the Heun's formulation)

% Stability check (diffusion)
dt_max = dx^2/(2*p.D);
fprintf('dx = %.3e, dt = %.3e, dt_max(diffusion) ~ %.3e\n', dx, dt, dt_max);
if dt > dt_max
    warning('Heun is explicit too; if dt > dx^2/(2D), instability may appear. Consider increasing Nt.');
end

%% Initial condition
c0 = ic_gaussian(x,p);
    

%% ==========================================================
%% (A) Reference solution in space: PDEPE
%% ==========================================================
m = 0;
sol_pdepe = pdepe(m, ...
    @(x,t,u,dudx) pdefun_reacdiff(x,t,u,dudx,p), ...
    @(x)          ic_gaussian(x,p), ...
    @(xl,ul,xr,ur,t) bcfun_neumann(xl,ul,xr,ur,t,p), ...
    x, t);

C_pdepe = sol_pdepe(:,:,1);   % Nt x Nx

%% ==========================================================
%% (B) Reference solution in time: MOL + ODE45
%% ==========================================================
opts = odeset('RelTol',1e-8,'AbsTol',1e-10);
[t_ode, C_ode] = ode45(@(tt,cc) rhs_MOL_reacdiff(cc,p,dx,Nx), t, c0, opts); %#ok<ASGLU>
% C_ode is Nt x Nx

%% ==========================================================
%% (C) Check: pdepe vs MOL+ode45 (error should be ~0)
%% ==========================================================
ref_err_inf = max(abs(C_pdepe - C_ode), [], 'all');
fprintf('Max difference between pdepe and MOL+ode45 (all x,t): %.3e\n', ref_err_inf);

%% ==========================================================
%% (D) Heun (Explicit trapezoidal / RK2) using MOL
%% ==========================================================
C_H = zeros(Nt, Nx);
C_H(1,:) = c0.';

for n = 1:Nt-1
    y_n = C_H(n,:).';  % current state (Nx x 1)

    % f1 = f(y_n, t_n)
    f1 = rhs_MOL_reacdiff(y_n, p, dx,Nx);

    % predictor (Euler): y* = y_n + dt * f1
    y_star = y_n + dt * f1;

    % f2 = f(y*, t_{n+1})
    f2 = rhs_MOL_reacdiff(y_star, p, dx,Nx);

    % corrector (Heun): y_{n+1} = y_n + dt/2 * (f1 + f2)
    y_next = y_n + (dt/2) * (f1 + f2);


    C_H(n+1,:) = y_next.';
end




%% ==========================================================
%% (E) ERRORS AND CALCULATION OF NORMS
%%     Reference = MOL + ode45
%% ==========================================================

% ----------------------------------------------------------
% Absolute error point per point
% ----------------------------------------------------------
Err = abs(C_H - C_ode);     % Nt x Nx

% ----------------------------------------------------------
% (1) Absolute error in space (t = T)
% ----------------------------------------------------------
e_space = Err(end,:);       % 1 x Nx

figure;
plot(x, e_space, 'LineWidth', 2);
grid on; box on;
xlabel('x (mm)');
ylabel('|error(x,T)|');
title('Error absoluto punto a punto en el espacio (t = T)');

% ----------------------------------------------------------
% (2) Absolute error in TIME (x = L/2)
% ----------------------------------------------------------
x0 = L/2;
[~, ix0] = min(abs(x - x0));

e_time = Err(:,ix0);        % Nt x 1


figure;
plot(t, e_time, 'LineWidth', 2);
grid on; box on;
xlabel('t (days)');
ylabel('|error(x0,t)|');
title(sprintf('Error absoluto punto a punto en el tiempo (x = %.2f ≈ L/2)', x(ix0)));

% ----------------------------------------------------------
% (3) Calculation of norms (l2, l_inf, local absolute average error,
% relative average error
% ----------------------------------------------------------

% (1) Global L2 error over space-time  (Eq. 18)
err_global_L2 = sqrt( sum(Err(:).^2) * dx * dt );

% (2) Global Linf error over space-time (Eq. 19)
err_global_Linf = max(Err(:));

% (3) Mean absolute local error (pointwise mean)
err_mean_abs = mean(Err(:));

% (4) Mean relative local error (pointwise mean of relative error)
eps_rel = 1e-14;   % avoid division by zero
err_mean_rel = mean( Err(:) ./ (abs(C_ode(:)) + eps_rel) );

fprintf('\n===== ERROR METRICS vs ode45 =====\n');
fprintf('Global error Linf (x,t) : %.4e\n', err_global_Linf);
fprintf('Global error L2   (x,t) : %.4e\n', err_global_L2);
fprintf('Mean relative error      : %.4e\n', err_mean_rel);
fprintf('Mean absolute error      : %.4e\n', err_mean_abs);

% ----------------------------------------------------------
% (4) Representation of the 3D error
% ----------------------------------------------------------

[X,T] = meshgrid(x, t);
figure;
surf(X, T, Err, 'EdgeColor', 'none');
view(45,30);
colormap turbo;
colorbar;

xlabel('x (mm)');
ylabel('t (days)');
zlabel('|error|');
title('Absolute error in space and time');

%% ==========================================================
%                 LOCAL FUNCTIONS
%% ==========================================================

function u0 = ic_gaussian(x,p)
u0 = p.A * exp(-(x - p.x0).^2 / (2*p.sigma^2));
end

function [c,f,s] = pdefun_reacdiff(x,t,u,dudx,p) %#ok<INUSD>
c = 1;
f = p.D * dudx;
s = p.rho * u * (1 - u/p.cmax);
end

function [pl,ql,pr,qr] = bcfun_neumann(xl,ul,xr,ur,t,p) %#ok<INUSD,INUSD,INUSD,INUSD>
pl = 0; ql = 1;
pr = 0; qr = 1;
end

function dcdt = rhs_MOL_reacdiff(c,p,dx,Nx)
% Semi-discrete MOL RHS using 2nd-order FD + Neumann BC

cxx = zeros(Nx,1);

% interior
cxx(2:Nx-1) = (c(1:Nx-2) - 2*c(2:Nx-1) + c(3:Nx)) / dx^2;

% Neumann via mirroring
cxx(1)  = 2*(c(2)    - c(1))  / dx^2;
cxx(Nx) = 2*(c(Nx-1) - c(Nx)) / dx^2;

reaction = p.rho * c .* (1 - c/p.cmax);
dcdt = p.D * cxx + reaction;
end

function cxx = laplacian_neumann(c, dx)
% Same Laplacian used in MOL
Nx = numel(c);
cxx = zeros(Nx,1);
cxx(2:Nx-1) = (c(1:Nx-2) - 2*c(2:Nx-1) + c(3:Nx)) / dx^2;
cxx(1)  = 2*(c(2)    - c(1))  / dx^2;
cxx(Nx) = 2*(c(Nx-1) - c(Nx)) / dx^2;
end
