%% ==========================================================
% Forward Euler (explicit) vs Reference solutions
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

% Reference time grid (choose Nt = 20000 as in your figures)
Nt = 20000;
t  = linspace(0, T, Nt);
dt = t(2) - t(1);

% Stability check for Forward Euler (diffusion CFL)
dt_max = dx^2/(2*p.D);
fprintf('dx = %.3e, dt = %.3e, dt_max(diffusion) ~ %.3e\n', dx, dt, dt_max);
if dt > dt_max
    warning('Forward Euler may be unstable: dt > dx^2/(2D). Consider increasing Nt.');
end

%% Initial condition (same for all solvers)
c0 = ic_gaussian(x,p).';   % column vector

%% ==========================================================
%% (A) Reference solution in space: PDEPE
%% ==========================================================
m = 0;
sol_pdepe = pdepe(m, ...
    @(x,t,u,dudx) pdefun_reacdiff(x,t,u,dudx,p), ...
    @(x)          ic_gaussian(x,p), ...
    @(xl,ul,xr,ur,t) bcfun_neumann(xl,ul,xr,ur,t,p), ...
    x, t);

C_pdepe = sol_pdepe(:,:,1);   % size Nt x Nx

%% ==========================================================
%% (B) Reference solution in time: MOL + ODE45
%% ==========================================================
rhs = @(t,c) rhs_MOL_reacdiff(c,p,dx,Nx);

opts = odeset('RelTol',1e-8,'AbsTol',1e-10);
[t_ode, C_ode] = ode45(@(tt,cc) rhs(tt,cc), t, c0, opts); %#ok<ASGLU>
% C_ode is Nt x Nx (evaluated exactly at times 't')

%% ==========================================================
%% (C) Check: pdepe vs MOL+ode45 (error should be ~0)
%% ==========================================================
diff_ref = C_pdepe - C_ode;
ref_err_inf = max(abs(diff_ref), [], 'all');
fprintf('Max difference between pdepe and MOL+ode45 (all x,t): %.3e\n', ref_err_inf);

%% ==========================================================
%% (D) Forward Euler (explicit) using MOL
%% ==========================================================
C_FE = zeros(Nt, Nx);
C_FE(1,:) = c0.';

for n = 1:Nt-1
    c = C_FE(n,:).';  % column

    cxx = laplacian_neumann(c, dx);
    reaction = p.rho * c .* (1 - c/p.cmax);

    c_next = c + dt * (p.D*cxx + reaction);

    % Optional safety: avoid negative values due to numerical oscillations
    c_next = max(c_next, 0);

    C_FE(n+1,:) = c_next.';
end

%% (E) ABSOLUTE POINTWISE ERRORS + L_inf and L2 norms
%%     Reference = MOL + ode45
%% ==========================================================

% ----------------------------------------------------------
% Error absoluto punto a punto
% ----------------------------------------------------------
Err = abs(C_FE - C_ode);     % Nt x Nx

% ----------------------------------------------------------
% (1) Error punto a punto en el ESPACIO (t = T)
% ----------------------------------------------------------
e_space = Err(end,:);       % 1 x Nx

% Normas espaciales (t = T)
err_space_inf = max(e_space);
err_space_L2  = sqrt( sum(e_space.^2) * dx );

fprintf('\nERROR ESPACIAL (t = T) – Forward Euler vs ode45\n');
fprintf('  ||error||_inf = %.3e\n', err_space_inf);
fprintf('  ||error||_L2  = %.3e\n', err_space_L2);

figure;
plot(x, e_space, 'LineWidth', 2);
grid on; box on;
xlabel('x (mm)');
ylabel('|error(x,T)|');
title('Error absoluto punto a punto en el espacio (t = T)');

% ----------------------------------------------------------
% (2) Error punto a punto en el TIEMPO (x = L/2)
% ----------------------------------------------------------
x0 = L/2;
[~, ix0] = min(abs(x - x0));

e_time = Err(:,ix0);        % Nt x 1

% Normas temporales (x = L/2)
err_time_inf = max(e_time);
err_time_L2  = sqrt( sum(e_time.^2) * dt );

fprintf('\nERROR TEMPORAL (x = L/2) – Forward euler vs ode45\n');
fprintf('  ||error||_inf = %.3e\n', err_time_inf);
fprintf('  ||error||_L2  = %.3e\n', err_time_L2);

figure;
plot(t, e_time, 'LineWidth', 2);
grid on; box on;
xlabel('t (days)');
ylabel('|error(x0,t)|');
title(sprintf('Error absoluto punto a punto en el tiempo (x = %.2f ≈ L/2)', x(ix0)));







%% ==========================================================
%                 LOCAL FUNCTIONS
%% ==========================================================

function u0 = ic_gaussian(x,p)
% Gaussian inoculum centered at L/2

u0 = p.A * exp(-(x - p.x0).^2 / (2*p.sigma^2));
end

function [c,f,s] = pdefun_reacdiff(x,t,u,dudx,p) %#ok<INUSD>
% pdepe form: c*u_t = d/dx(f) + s
c = 1;
f = p.D * dudx;
s = p.rho * u * (1 - u/p.cmax);
end

function [pl,ql,pr,qr] = bcfun_neumann(xl,ul,xr,ur,t,p) %#ok<INUSD,INUSD,INUSD,INUSD>
% Neumann no-flux: f = 0 -> p=0, q=1
pl = 0; ql = 1;
pr = 0; qr = 1;
end

function dcdt = rhs_MOL_reacdiff(c,p,dx,Nx)
% MOL semi-discrete RHS: c_t = D*cxx + reaction
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
