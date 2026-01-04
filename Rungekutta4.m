clear; close all; clc;

L = 200.0;      % [mm]
T = 30.0;      % [days]
p.L = L;

p.D    = 0.126; 
p.rho  = 1.0; 
p.cmax = 1.0;

p.A = 0.1;
p.x0 = p.L / 2;
p.sigma = 10;

Nx = 800;
x  = linspace(0, L, Nx);
dx = x(2) - x(1); 

Nt = 20000;
t  = linspace(0, T, Nt);
dt = t(2) - t(1); 

dt_max = dx^2/(2*p.D);
fprintf('dx = %.3e, dt = %.3e, dt_max(diffusion) ~ %.3e\n', dx, dt, dt_max);
if dt > dt_max
    warning('Forward Euler may be unstable: dt > dx^2/(2D). Consider increasing Nt.');
end


c0 = ic_gaussian(x,p).';   

%% (A) Reference solution in space: PDEPE
m = 0;
sol_pdepe = pdepe(m, ...
    @(x,t,u,dudx) pdefun_reacdiff(x,t,u,dudx,p), ...
    @(x)          ic_gaussian(x,p), ...
    @(xl,ul,xr,ur,t) bcfun_neumann(xl,ul,xr,ur,t,p), ...
    x, t);

C_pdepe = sol_pdepe(:,:,1);   

%% (B) Reference solution in time: MOL + ODE45
rhs = @(t,c) rhs_MOL_reacdiff(c,p,dx,Nx);

opts = odeset('RelTol',1e-8,'AbsTol',1e-10);
[t_ode, C_ode] = ode45(@(tt,cc) rhs(tt,cc), t, c0, opts); %#ok<ASGLU>

%% (C) Check: pdepe vs MOL+ode45 (error should be ~0)
diff_ref = C_pdepe - C_ode;
ref_err_inf = max(abs(diff_ref), [], 'all');
fprintf('Max difference between pdepe and MOL+ode45 (all x,t): %.3e\n', ref_err_inf);

%% (D) ERK4 (Runge–Kutta explicit of 4th order) using MOL
C_FE = zeros(Nt, Nx);
C_FE(1,:) = c0.';

for n = 1:Nt-1
    c = C_FE(n,:).';  

    k1 = rhs_MOL_reacdiff(c, p, dx, Nx);
    k2 = rhs_MOL_reacdiff(c + 0.5*dt*k1, p, dx, Nx);
    k3 = rhs_MOL_reacdiff(c + 0.5*dt*k2, p, dx, Nx);
    k4 = rhs_MOL_reacdiff(c + dt*k3, p, dx, Nx);

    c_next = c + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);
    c_next = max(c_next, 0);

    C_FE(n+1,:) = c_next.';
end

%% (E) ERRORS AND CALCULATION OF NORMS

Err = abs(C_FE - C_ode);   
e_space = Err(end,:);       

figure;
plot(x, e_space, 'LineWidth', 2);
grid on; box on;
xlabel('x (mm)');
ylabel('|error(x,T)|');
title('Absolute error in space (t = T)');

x0 = L/2;
[~, ix0] = min(abs(x - x0));

e_time = Err(:,ix0);     

figure;
plot(t, e_time, 'LineWidth', 2);
grid on; box on;
xlabel('t (days)');
ylabel('|error(x0,t)|');
title(sprintf('Absolute error in time (x = %.2f ≈ L/2)', x(ix0)));


err_global_L2 = sqrt( sum(Err(:).^2) * dx * dt );

err_global_Linf = max(Err(:));

err_mean_abs = mean(Err(:));

eps_rel = 1e-14;   
err_mean_rel = mean( Err(:) ./ (abs(C_ode(:)) + eps_rel) );

fprintf('\n===== ERROR METRICS vs ode45 =====\n');
fprintf('Global error Linf (x,t) : %.4e\n', err_global_Linf);
fprintf('Global error L2   (x,t) : %.4e\n', err_global_L2);
fprintf('Mean relative error      : %.4e\n', err_mean_rel);
fprintf('Mean absolute error      : %.4e\n', err_mean_abs);


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


%                 LOCAL FUNCTIONS

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
cxx = zeros(Nx,1);

cxx(2:Nx-1) = (c(1:Nx-2) - 2*c(2:Nx-1) + c(3:Nx)) / dx^2;

cxx(1)  = 2*(c(2)    - c(1))  / dx^2;
cxx(Nx) = 2*(c(Nx-1) - c(Nx)) / dx^2;

reaction = p.rho * c .* (1 - c/p.cmax);
dcdt = p.D * cxx + reaction;
end

function cxx = laplacian_neumann(c, dx)
Nx = numel(c);
cxx = zeros(Nx,1);
cxx(2:Nx-1) = (c(1:Nx-2) - 2*c(2:Nx-1) + c(3:Nx)) / dx^2;
cxx(1)  = 2*(c(2)    - c(1))  / dx^2;
cxx(Nx) = 2*(c(Nx-1) - c(Nx)) / dx^2;
end
