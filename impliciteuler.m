%% ==========================================================
% Backward Euler (Euler Indirecto, implícito) vs Reference solutions
%
% PDE: c_t = D c_xx + rho c (1 - c/cmax)
% BC:  Neumann (no-flux)
% IC:  Gaussian inoculum centered at x=L/2
%
% Strategy:
%  - Spatial reference: pdepe  -> compare c(x,T)
%  - Temporal reference: MOL + ode45 -> compare c(x0,t)
%  - Implicit method: Backward Euler solved by Newton
%% ==========================================================

clear; close all; clc;

%% ------------------- Parameters ---------------------------
L = 200.0;      % [mm]
T = 30.0;       % [days]
p.L = L;

% Equation parameters
p.D    = 0.126; % [mm^2/days]
p.rho  = 1.0;   % [1/days]  
p.cmax = 1.0;

% Initial gaussian
p.A = 0.1;
p.x0 = p.L/2;
p.sigma = 10;

Nx = 800;
x  = linspace(0, L, Nx);
dx = x(2) - x(1);

% For this method we can use a few Nt but we mantain 20000 to compare with
% the other numerical methods
Nt = 20000;              
t  = linspace(0, T, Nt);
dt = t(2) - t(1);

fprintf('dx = %.3e, dt = %.3e\n', dx, dt);

%% Initial condition
c0 = ic_gaussian(x,p).';   % column (Nx x 1)

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

%% (C) Check: pdepe vs MOL+ode45
ref_err_inf = max(abs(C_pdepe - C_ode), [], 'all');
fprintf('Max difference between pdepe and MOL+ode45 (all x,t): %.3e\n', ref_err_inf);

%% ==========================================================
%% (D) Backward Euler (Implicit) using Newton
%% ==========================================================
% Build Laplacian matrix with Neumann BC: L*c approx c_xx
Lmat = laplacian_matrix_neumann(Nx, dx);   % sparse Nx x Nx
I = speye(Nx); %d/dc(c)

C_BE = zeros(Nt, Nx);
C_BE(1,:) = c0.';

newton_tol   = 1e-10;
newton_maxit = 20;

for n = 1:Nt-1
    c_n = C_BE(n,:).';   % known state (Nx x 1)

    % Initial guess:
    %  - simplest: c_{n+1}^{(0)} = c_n
    %  - better: one explicit Euler predictor
    f_n = p.D*(Lmat*c_n) + p.rho*c_n.*(1 - c_n/p.cmax);
    c = c_n + dt*f_n;  % predictor

    % Newton iterations to solve:
    % G(c) = c - c_n - dt*( D*L*c + rho*c.*(1-c/cmax) ) = 0
    for k = 1:newton_maxit
        % Nonlinear term and its derivative
        R  = p.rho * c .* (1 - c/p.cmax);                  % reaction
        dR = p.rho * (1 - 2*c/p.cmax);                     % d/dc reaction (pointwise)

        G = c - c_n - dt*( p.D*(Lmat*c) + R );            % residual (must be equal to zero or < that some tolerance)

        if norm(G, inf) < newton_tol
            break;
        end

        % Jacobian: J = I - dt*( D*L + diag(dR) )
        J = I - dt*( p.D*Lmat + spdiags(dR, 0, Nx, Nx) ); % J = c' - dt(DL+diag(DR))

        % Newton step -> resolves J*delta = G and updates c = c-delta
        % (searches the vector delta that, multiplied by the J produce the
        % error G)
         
        delta = J \ G;
        c = c - delta; % update c

        % (Optional) stop if update is tiny
        if norm(delta, inf) < 1e-12
            break;
        end
    end

    % Optional positivity clamp (same idea as in tu explícito)
    c = max(c, 0);

    C_BE(n+1,:) = c.';
end

%% ==========================================================
%% (E) ERRORS AND NORMS (Reference = MOL + ode45)
%% ==========================================================
Err = abs(C_BE - C_ode);  % Nt x Nx

% (1) Absolute error in SPACE (t = T)
e_space = Err(end,:);

figure;
plot(x, e_space, 'LineWidth', 2);
grid on; box on;
xlabel('x (mm)');
ylabel('|error(x,T)|');
title('Backward Euler: absolute error in space (t = T)');

% (2) Absolute error in TIME (x = L/2)
x0 = L/2;
[~, ix0] = min(abs(x - x0));
e_time = Err(:, ix0);

figure;
plot(t, e_time, 'LineWidth', 2);
grid on; box on;
xlabel('t (days)');
ylabel('|error(x0,t)|');
title(sprintf('Backward Euler: absolute error in time (x = %.2f ≈ L/2)', x(ix0)));

% Norms
err_global_L2   = sqrt( sum(Err(:).^2) * dx * dt );
err_global_Linf = max(Err(:));
err_mean_abs    = mean(Err(:));
eps_rel = 1e-14;
err_mean_rel    = mean( Err(:) ./ (abs(C_ode(:)) + eps_rel) );

fprintf('\n===== BACKWARD EULER ERROR METRICS vs ode45 =====\n');
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
cxx = zeros(Nx,1);
cxx(2:Nx-1) = (c(1:Nx-2) - 2*c(2:Nx-1) + c(3:Nx)) / dx^2;
cxx(1)  = 2*(c(2)    - c(1))  / dx^2;      % Neumann mirror
cxx(Nx) = 2*(c(Nx-1) - c(Nx)) / dx^2;

reaction = p.rho * c .* (1 - c/p.cmax);
dcdt = p.D * cxx + reaction;
end

function L = laplacian_matrix_neumann(Nx, dx)
% Builds sparse matrix L such that (L*c) approximates c_xx with Neumann BC

e = ones(Nx,1);

% Base tridiagonal for interior: (1, -2, 1)/dx^2
L = spdiags([e -2*e e], -1:1, Nx, Nx) / dx^2;

% Neumann adjustments at boundaries using mirror:
% cxx(1)  = 2*(c2 - c1)/dx^2  -> row1: [-2,  2, 0, ...]/dx^2
% cxx(Nx) = 2*(c_{Nx-1} - c_{Nx})/dx^2 -> last row: [..., 2, -2]/dx^2
L(1,1) = -2/dx^2;  L(1,2) =  2/dx^2;
L(Nx,Nx) = -2/dx^2; L(Nx,Nx-1) = 2/dx^2;
end
