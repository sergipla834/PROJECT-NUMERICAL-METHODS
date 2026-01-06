clear; close all; clc;

L = 200.0;      % mm
T = 30.0;       % days

p.L = L;
p.D    = 0.1;  
p.rho  = 1.0;   
p.cmax = 1.0;

p.A = 0.1;
p.x0 = L/2;
p.sigma = 10;

Nx = 800;
x  = linspace(0, L, Nx);
dx = x(2) - x(1);

Nt = 2500;            
t  = linspace(0, T, Nt);
dt = t(2) - t(1);

fprintf('dx = %.3e, dt = %.3e, Nx=%d, Nt=%d\n', dx, dt, Nx, Nt);

c0 = ic_gaussian(x,p).';   


%% (A) Reference solution: PDEPE
m = 0;
sol_pdepe = pdepe(m, ...
    @(xx,tt,u,dudx) pdefun_reacdiff(xx,tt,u,dudx,p), ...
    @(xx)          ic_gaussian(xx,p), ...
    @(xl,ul,xr,ur,tt) bcfun_neumann(xl,ul,xr,ur,tt,p), ...
    x, t);

C_pdepe = sol_pdepe(:,:,1);   


%% (B) Reference solution: MOL + ODE45
opts = odeset('RelTol',1e-8,'AbsTol',1e-10);
[~, C_ode] = ode45(@(tt,cc) rhs_MOL_reacdiff(cc,p,dx,Nx), t, c0, opts);

ref_diff = max(abs(C_pdepe - C_ode), [], 'all');
fprintf('Max |pdepe - MOL(ode45)| over all (x,t): %.3e\n', ref_diff);


%% (C) ROSENBROCK1 METHOD (ONLY NUMERICAL PART CHANGES)
Lmat = laplacian_matrix_neumann(Nx, dx); 
I = speye(Nx);

C_RB1 = zeros(Nt, Nx);
C_RB1(1,:) = c0.';

gamma = 1 - 1/sqrt(2);   

for n = 1:Nt-1
    c_n = C_RB1(n,:).';

    Rn = p.rho * c_n .* (1 - c_n/p.cmax);
    fn = p.D*(Lmat*c_n) + Rn;

    dR = p.rho * (1 - 2*c_n/p.cmax);
    Jn = p.D*Lmat + spdiags(dR, 0, Nx, Nx);

    A = (I - gamma*dt*Jn);
    k = A \ fn;

    c_np1 = c_n + dt*k;

    c_np1 = max(c_np1, 0);

    C_RB1(n+1,:) = c_np1.';
end


%% (D) Errors vs MOL reference (ode45)
Err = abs(C_RB1 - C_ode); 

e_space = Err(end,:);
figure;
plot(x, e_space, 'LineWidth', 2);
grid on; box on;
xlabel('x (mm)');
ylabel('|error(x,T)|');
title('Rosenbrock1: absolute error in space (t = T)');

x0 = L/2;
[~, ix0] = min(abs(x - x0));
e_time = Err(:, ix0);
figure;
plot(t, e_time, 'LineWidth', 2);
grid on; box on;
xlabel('t (days)');
ylabel('|error(x0,t)|');
title(sprintf('Rosenbrock1: absolute error in time (x = %.2f ≈ L/2)', x(ix0)));

[X,Tm] = meshgrid(x, t);
figure;
surf(X, Tm, Err, 'EdgeColor','none');
view(45,30);
colormap turbo; colorbar;
xlabel('x (mm)');
ylabel('t (days)');
zlabel('|error|');
title('Rosenbrock1: absolute error |e(x,t)|');


%% (E) ERRORS AND CALCULATIONS OF NORMS

err_global_L2 = sqrt( sum(Err(:).^2) * dx * dt );
err_global_Linf = max(Err(:));
err_mean_abs = mean(Err(:));

eps_rel = 1e-14;   
err_mean_rel = mean( Err(:) ./ (abs(C_ode(:)) + eps_rel) );

err_space_Linf = max(e_space);
err_space_L2   = sqrt( sum(e_space.^2) * dx );

err_time_Linf = max(e_time);
err_time_L2   = sqrt( sum(e_time.^2) * dt );

fprintf('\n===== ROSENBROCK1 ERROR METRICS vs ode45 =====\n');
fprintf('Global error Linf (x,t) : %.4e\n', err_global_Linf);
fprintf('Global error L2   (x,t) : %.4e\n', err_global_L2);
fprintf('Mean relative error      : %.4e\n', err_mean_rel);
fprintf('Mean absolute error      : %.4e\n', err_mean_abs);

fprintf('\nAt t = T (space slice):\n');
fprintf('  Linf: %.4e\n', err_space_Linf);
fprintf('  L2  : %.4e\n', err_space_L2);

fprintf('\nAt x = L/2 (time slice):\n');
fprintf('  Linf: %.4e\n', err_time_Linf);
fprintf('  L2  : %.4e\n', err_time_L2);


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

function L = laplacian_matrix_neumann(Nx, dx)
e = ones(Nx,1);
L = spdiags([e -2*e e], -1:1, Nx, Nx) / dx^2;
L(1,1) = -2/dx^2;  L(1,2) =  2/dx^2;
L(Nx,Nx) = -2/dx^2; L(Nx,Nx-1) = 2/dx^2;
end
